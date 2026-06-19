import re
import logging

logger = logging.getLogger("FAANG_RAG_CORE")

PROHIBITED_PATTERNS = [
    r"(?i)\b(hack|bypass|override|ignore instructions|jailbreak|delete database|drop table)\b",
    r"(?i)\b(malware|virus|exploit|ransomware|trojan)\b",
    r"(?i)\b(generate insult|hate speech|offensive)\b"
]

class SecurityGuardrail:
    @staticmethod
    def verify_query_safety(query: str) -> bool:
        for pattern in PROHIBITED_PATTERNS:
            if re.search(pattern, query):
                logger.warning(f"SECURITY_ALERT: Inbound transaction blocked by pattern matching filter: '{pattern}'")
                return False
        return True
