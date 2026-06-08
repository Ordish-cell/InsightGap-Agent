"""Email provider – mock and optionally SMTP.

Mock provider writes audit logs only, never sends real email.
SMTP provider uses environment variables and must never log passwords.
"""

import logging
import os
import smtplib
from email.mime.text import MIMEText
from typing import Any

logger = logging.getLogger(__name__)


class MockEmailProvider:
    """Mock email provider that only writes audit logs, never sends."""

    async def send_email(self, to: str, subject: str, body: str) -> dict[str, Any]:
        return {
            "provider": "mock",
            "sent": False,
            "message": "EMAIL_PROVIDER=mock, email was not actually sent.",
            "to": to,
            "subject": subject,
            "body_preview": body[:200],
        }


class SMTPEmailProvider:
    """Real SMTP email provider. Credentials from environment variables only."""

    def __init__(self) -> None:
        self.host = os.getenv("SMTP_HOST", "")
        self.port = int(os.getenv("SMTP_PORT", "587"))
        self.username = os.getenv("SMTP_USERNAME", "")
        self.password = os.getenv("SMTP_PASSWORD", "")
        self.use_tls = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
        self.from_email = os.getenv("EMAIL_FROM", self.username)

    def _smtp_params(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "username": self.username,
            "use_tls": self.use_tls,
            # NEVER include password in any log output
        }

    async def send_email(self, to: str, subject: str, body: str) -> dict[str, Any]:
        if not self.host or not self.username:
            return {
                "provider": "smtp",
                "sent": False,
                "message": "SMTP not configured. Set SMTP_HOST and SMTP_USERNAME.",
                "smtp_params": self._smtp_params(),
                "to": to,
                "subject": subject,
            }

        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = self.from_email
        msg["To"] = to

        try:
            server = smtplib.SMTP(self.host, self.port, timeout=15)
            if self.use_tls:
                server.starttls()
            server.login(self.username, self.password)
            server.sendmail(self.from_email, [to], msg.as_string())
            server.quit()
            return {
                "provider": "smtp",
                "sent": True,
                "message": "Email sent successfully.",
                "to": to,
                "subject": subject,
            }
        except Exception as exc:
            logger.error("SMTP send failed: %s", exc)
            return {
                "provider": "smtp",
                "sent": False,
                "message": f"SMTP send failed. Error: {exc}",
                "to": to,
                "subject": subject,
            }


def get_email_provider() -> MockEmailProvider | SMTPEmailProvider:
    provider = os.getenv("EMAIL_PROVIDER", "mock").lower()
    if provider == "smtp":
        return SMTPEmailProvider()
    return MockEmailProvider()
