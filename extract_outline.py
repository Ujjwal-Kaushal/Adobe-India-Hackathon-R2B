# extract_outline.py

import fitz  # PyMuPDF
import json
import re
from pathlib import Path
from collections import defaultdict
import logging
import statistics

class PDFOutlineExtractor:
    def __init__(self):
        # We can disable the logger for the main application to avoid verbose console output
        # during the run, as the main script provides its own progress updates.
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.WARNING) # Set to WARNING to hide INFO messages
        
        # Regex to find standalone dates to exclude them from headings
        self.date_pattern = re.compile(
            r'^(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2},?\s+\d{4}$|'
            r'^\d{1,2}\s+(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\s+\d{4}$',
            re.IGNORECASE
        )
        # This pattern uses common structural words ("Chapter", "Appendix") alongside a generic
        # numbering pattern (\d+(\.\d+)*) to reliably identify headings.
        self.numbered_heading_pattern = re.compile(r'^(\d+(\.\d+)*)\s+.*|^(Chapter\s+\d+):.*|^(Appendix\s+[A-Z]):.*', re.IGNORECASE)
        
        # These are configurable heuristics, not hardcoded content values.
        self.min_heading_length = 3
        self.max_heading_length = 250 

    def analyze_text_properties(self, doc):
        """Analyze text properties to determine the most common (body) font size."""
        font_sizes = defaultdict(int)
        
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            font_sizes[round(span["size"])] += len(span["text"].strip())

        body_font_size = 10.0 # A sensible default if analysis fails
        if font_sizes:
            # The font size with the most characters is likely the body text
            body_font_size = float(max(font_sizes.items(), key=lambda item: item[1])[0])
            self.logger.info(f"Detected body font size: {body_font_size}")

        return body_font_size

    def identify_footers(self, doc, page_threshold=0.5):
        """Identify recurring text at the bottom of pages to exclude it from content."""
        footer_candidates = defaultdict(int)
        num_pages = len(doc)
        
        for page in doc:
            # Check the bottom 15% of the page for potential footers
            footer_zone = fitz.Rect(0, page.rect.height * 0.85, page.rect.width, page.rect.height)
            text = page.get_text(clip=footer_zone).strip()
            if text:
                # Normalize footer text by removing page numbers and common footer terms
                normalized_text = re.sub(r'\d+$', '', text, flags=re.IGNORECASE).strip()
                normalized_text = re.sub(r'page\s*\d+\s*of\s*\d+', '', normalized_text, flags=re.IGNORECASE).strip()
                if len(normalized_text) > 5:
                    footer_candidates[normalized_text] += 1
        
        # A footer is text that appears on at least half the pages.
        footers = {text for text, count in footer_candidates.items() if count >= num_pages * page_threshold and len(text) > 10}
        self.logger.info(f"Identified potential footers: {footers}")
        return footers

    def _clean_title_text(self, text):
        """Clean title text by removing garbled, repeated patterns."""
        text = re.sub(r'\s+', ' ', text).strip()
        words = text.split()
        if len(words) < 4: return text

        pattern_to_check = " ".join(words[:3])
        if text.count(pattern_to_check) > 2:
            for i in range(len(words)):
                if not " ".join(words[i:]).startswith(pattern_to_check):
                    return " ".join(words[i:])
        return text

    def extract_title(self, doc):
        """Extract document title from the first page, merging adjacent large, centered text blocks."""
        first_page = doc[0]
        blocks = first_page.get_text("dict", sort=True)["blocks"]
        page_width = first_page.rect.width
        
        candidates = []
        for block in filter(lambda b: b['bbox'][1] < first_page.rect.height / 2, blocks):
            if "lines" in block:
                bbox = block["bbox"]
                text = " ".join("".join(span["text"] for span in line["spans"]).strip() for line in block["lines"]).strip()
                if not text or len(text) < 5: continue
                
                spans = [span for line in block["lines"] for span in line["spans"]]
                if not spans: continue

                avg_font_size = statistics.mean(s['size'] for s in spans)
                is_bold = any("Bold" in s['font'] for s in spans)
                left_margin = bbox[0]
                right_margin = page_width - bbox[2]
                is_centered = abs(left_margin - right_margin) < page_width * 0.25
                
                if avg_font_size > 14 and (is_centered or is_bold):
                     candidates.append({'text': text, 'size': avg_font_size, 'y0': bbox[1]})

        if not candidates: return ""
        candidates.sort(key=lambda x: x['y0'])

        title_parts = [candidates[0]['text']]
        last_y = candidates[0]['y0']
        
        for cand in candidates[1:]:
            if abs(cand['y0'] - last_y) < 60:
                 title_parts.append(cand['text'])
                 last_y = cand['y0']
            else:
                 break 
        
        full_title = " ".join(title_parts)
        return self._clean_title_text(full_title)

    def is_heading(self, block, body_font_size, page_width, footers):
        """Determine if a text block is a heading based on a combination of heuristics."""
        if "lines" not in block or not block["lines"]: return False

        full_text = " ".join("".join(span["text"] for span in line["spans"]).strip() for line in block["lines"]).strip()
        full_text = re.sub(r'\s+', ' ', full_text)

        if not full_text or len(full_text) < self.min_heading_length or len(full_text) > self.max_heading_length: return False
        if any(footer in full_text for footer in footers): return False
        if self.date_pattern.match(full_text) or full_text.isdigit(): return False
        if full_text.endswith('.') and not full_text.isupper(): return False

        spans = [span for line in block["lines"] for span in line["spans"]]
        if not spans: return False
        
        avg_font_size = statistics.mean(s['size'] for s in spans)
        is_bold = any("Bold" in s['font'] or "Black" in s['font'] or (s['flags'] & 2**4) for s in spans)
        is_all_caps = full_text.isupper() and len(full_text) > 4
        
        bbox = block["bbox"]
        left_indent = bbox[0]
        right_space = page_width - bbox[2]
        
        is_larger_than_body = avg_font_size > body_font_size * 1.1 
        is_left_aligned_short_line = left_indent < (page_width * 0.2) and right_space > (page_width * 0.3)
        
        if is_left_aligned_short_line and (is_bold or is_larger_than_body): return True
        if is_all_caps and (is_bold or is_larger_than_body): return True
        if is_bold and is_larger_than_body: return True
        if self.numbered_heading_pattern.match(full_text): return True

        return False

    def get_heading_level(self, heading, font_size_tiers, numbered_level):
        """Assigns H1, H2, etc. based on font size tiers and numbered prefixes."""
        if numbered_level > 0: return f"H{numbered_level}"

        size = heading['size']
        if not font_size_tiers: return "H2"
        
        level = len(font_size_tiers)
        for i, tier_size in enumerate(font_size_tiers):
            if size >= tier_size * 0.98:
                level = i + 1
                break
        
        return f"H{level}"

    def normalize_hierarchy(self, outline):
        """Ensures the heading hierarchy is logical (e.g., no H3 directly after an H1)."""
        if not outline: return []
        
        last_level_num = 0
        normalized_outline = []

        for item in outline:
            current_level_num = int(item['level'][1:])
            
            if current_level_num > last_level_num + 1:
                current_level_num = last_level_num + 1
            
            item['level'] = f"H{current_level_num}"
            last_level_num = current_level_num
            normalized_outline.append(item)

        return normalized_outline

    def extract_outline(self, pdf_path):
        """Main function to extract a structured outline from a PDF."""
        try:
            with fitz.open(pdf_path) as doc:
                if doc.is_form_pdf: 
                    self.logger.warning(f"'{Path(pdf_path).name}' detected as a form. Extracting title only.")
                    return {"title": self.extract_title(doc), "outline": []}

                body_font_size = self.analyze_text_properties(doc)
                footers = self.identify_footers(doc)
                title = self.extract_title(doc)

                headings = []
                for page_num, page in enumerate(doc):
                    blocks = page.get_text("dict", sort=True)["blocks"]
                    for block in blocks:
                        if self.is_heading(block, body_font_size, page.rect.width, footers):
                            text = " ".join("".join(s["text"] for s in l["spans"]) for l in block["lines"]).strip()
                            text = re.sub(r'\s+', ' ', text)
                            
                            spans = [s for l in block['lines'] for s in l['spans']]
                            if not spans: continue
                            avg_font_size = statistics.mean(s['size'] for s in spans)
                            
                            headings.append({"text": text, "page": page_num + 1, "size": avg_font_size})

                if not headings:
                    return {"title": title, "outline": []}

                seen = set()
                unique_headings = [h for h in headings if (h["text"], h["page"]) not in seen and not seen.add((h["text"], h["page"]))]
                
                heading_sizes = sorted(list(set(round(h['size'], 1) for h in unique_headings)), reverse=True)
                font_size_tiers = []
                if heading_sizes:
                    font_size_tiers.append(heading_sizes[0])
                    for size in heading_sizes[1:]:
                        if font_size_tiers[-1] - size > 1.5:
                            if len(font_size_tiers) < 5:
                                font_size_tiers.append(size)
                self.logger.info(f"Detected heading font size tiers: {font_size_tiers}")

                outline = []
                for h in unique_headings:
                    numbered_level = 0
                    match = self.numbered_heading_pattern.match(h['text'])
                    if match:
                        if match.group(1):
                            numbered_level = match.group(1).count('.') + 1
                        else:
                            numbered_level = 1

                    level = self.get_heading_level(h, font_size_tiers, numbered_level)
                    outline.append({"level": level, "text": h["text"].strip(), "page": h["page"]})
                
                if outline and title and title.lower() in outline[0]['text'].lower():
                    outline[0]['level'] = "H1"

                final_outline = self.normalize_hierarchy(outline)

                return {"title": title, "outline": final_outline}

        except Exception as e:
            self.logger.error(f"Error processing {pdf_path}: {str(e)}")
            return {"title": "Error processing document", "outline": []}