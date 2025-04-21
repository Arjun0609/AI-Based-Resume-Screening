import re
import datetime
import numpy as np
import pandas as pd
from dateutil import parser
import logging

logger = logging.getLogger(__name__)


class EmploymentAnalyzer:
    def __init__(self):
        logger.info("Initializing EmploymentAnalyzer")

    def extract_employment_history(self, resume_text):
        date_patterns = [
            r"\b((?:19|20)\d{2})\s*[-–—]\s*((?:19|20)\d{2}|Present|Current|Now)\b",
            r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* (?:19|20)\d{2}\s*[-–—]\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* (?:19|20)\d{2}\b",
            r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* (?:19|20)\d{2}\s*[-–—]\s*(?:Present|Current|Now)\b",
            r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* (?:19|20)\d{2}\s*(to)\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* (?:19|20)\d{2}\b",
        ]

        combined_pattern = "|".join(date_patterns)

        matches = re.finditer(combined_pattern, resume_text)

        employment_records = []

        for match in matches:
            date_range = match.group(0)

            start_pos = max(0, match.start() - 100)
            end_pos = min(len(resume_text), match.end() + 100)
            context = resume_text[start_pos:end_pos]

            duration_info = self._parse_date_range(date_range)

            job_title = self._extract_job_title(context)
            company = self._extract_company(context)

            record = {
                "date_range": date_range,
                "start_date": duration_info.get("start_date"),
                "end_date": duration_info.get("end_date"),
                "duration_months": duration_info.get("duration_months"),
                "is_current": duration_info.get("is_current", False),
                "job_title": job_title,
                "company": company,
                "context": context,
            }

            employment_records.append(record)

        employment_records.sort(
            key=lambda x: (
                x["start_date"] if x["start_date"] else datetime.datetime(1900, 1, 1)
            ),
            reverse=True,
        )

        logger.info(f"Extracted {len(employment_records)} employment records")
        return employment_records

    def _parse_date_range(self, date_range):
        try:
            normalized_range = (
                date_range.strip().replace("–", "-").replace("—", "-").lower()
            )

            if " to " in normalized_range:
                parts = re.split(r"\s+to\s+", normalized_range, flags=re.IGNORECASE)
            elif "-" in normalized_range:
                parts = re.split(r"\s*-\s*", normalized_range)
            else:
                return {"error": "Invalid date range format"}

            if len(parts) != 2:
                return {"error": f"Unexpected format: {date_range}"}

            start_part, end_part = parts[0].strip(), parts[1].strip()

            is_current = end_part.lower() in ["present", "current", "now"]

            try:
                start_date = parser.parse(start_part)
            except Exception as e:
                logger.warning(f"Failed to parse start date '{start_part}': {str(e)}")
                return {"error": f"Failed to parse start date: {start_part}"}

            try:
                end_date = (
                    datetime.datetime.now() if is_current else parser.parse(end_part)
                )
            except Exception as e:
                logger.warning(f"Failed to parse end date '{end_part}': {str(e)}")
                return {"error": f"Failed to parse end date: {end_part}"}

            duration_months = (end_date.year - start_date.year) * 12 + (
                end_date.month - start_date.month
            )
            if duration_months < 0:
                duration_months = 0

            return {
                "start_date": start_date,
                "end_date": end_date,
                "duration_months": duration_months,
                "is_current": is_current,
            }

        except Exception as e:
            logger.error(f"Error parsing date range '{date_range}': {str(e)}")
            return {"error": str(e)}

    def _extract_job_title(self, context):
        job_title_patterns = [
            r"(Senior|Lead|Principal|Junior|Associate)?\s*(Software Engineer|Developer|Data Scientist|Manager|Director|Analyst|Administrator|Consultant|Specialist)",
            r"(Chief|VP|Head) of ([A-Z][a-z]+(?: [A-Z][a-z]+)*)",
            r"([A-Z][a-z]+) (Engineer|Manager|Director|Analyst|Administrator|Consultant|Specialist)",
        ]

        for pattern in job_title_patterns:
            match = re.search(pattern, context)
            if match:
                return match.group(0)

        return None

    def _extract_company(self, context):
        company_patterns = [
            r"at ((?:[A-Z][a-zA-Z]*\s*){1,3})",
            r"with ((?:[A-Z][a-zA-Z]*\s*){1,3})",
            r"for ((?:[A-Z][a-zA-Z]*\s*){1,3})",
        ]

        for pattern in company_patterns:
            match = re.search(pattern, context)
            if match:
                return match.group(1).strip()

        return None

    def analyze_employment_patterns(self, employment_records):
        if not employment_records:
            return {
                "average_tenure": 0,
                "total_experience": 0,
                "job_count": 0,
                "job_changing_frequency": 0,
                "has_gaps": False,
                "gap_details": [],
            }

        durations = [
            record["duration_months"]
            for record in employment_records
            if "duration_months" in record and record["duration_months"] is not None
        ]

        avg_tenure = np.mean(durations) if durations else 0
        total_experience = sum(durations)
        job_count = len(employment_records)

        years_of_experience = total_experience / 12 if total_experience > 0 else 1
        job_changing_frequency = (
            job_count / years_of_experience if years_of_experience > 0 else 0
        )

        gaps = self._detect_employment_gaps(employment_records)

        return {
            "average_tenure": avg_tenure,
            "average_tenure_years": avg_tenure / 12 if avg_tenure else 0,
            "total_experience": total_experience,
            "total_experience_years": total_experience / 12 if total_experience else 0,
            "job_count": job_count,
            "job_changing_frequency": job_changing_frequency,
            "has_gaps": len(gaps) > 0,
            "gap_count": len(gaps),
            "gap_details": gaps,
        }

    def _detect_employment_gaps(self, employment_records):
        """Detect gaps between employment periods."""
        if len(employment_records) < 2:
            return []

        sorted_records = sorted(
            [
                r
                for r in employment_records
                if r.get("end_date") and r.get("start_date")
            ],
            key=lambda x: x["end_date"],
        )

        gaps = []

        for i in range(len(sorted_records) - 1):
            current_end_date = sorted_records[i]["end_date"]
            next_start_date = sorted_records[i + 1]["start_date"]

            gap_months = (
                (next_start_date.year - current_end_date.year) * 12
                + (next_start_date.month - current_end_date.month)
                - 1
            )

            if gap_months > 3:
                gaps.append(
                    {
                        "start_date": current_end_date,
                        "end_date": next_start_date,
                        "duration_months": gap_months,
                    }
                )

        return gaps
