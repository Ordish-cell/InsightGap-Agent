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

# ── Tool aliases declared at provider registration point ──────────
# Registry collects and normalizes. Add canonical→aliases here.
EMAIL_TOOL_ALIASES: dict[str, list[str]] = {
    "email.send": [
        "发送邮件", "发邮件", "send_email", "email_send",
        "mail.send", "send.mail", "sendEmail", "寄邮件", "send_mail",
        "发一封", "给…发邮件",
    ],
    "email_mcp.create_draft": ["邮件草稿", "草稿", "draft", "email_draft"],
}


class MockEmailProvider:
    """Mock email provider that only writes audit logs, never sends."""

    async def send_email(self, to: str, subject: str, body: str) -> dict[str, Any]:
        return {
            "success": True,
            "provider": "mock",
            "sent": False,
            "to": to,
            "subject": subject,
            "body": body,
            "body_preview": body[:200],
            "message": "EMAIL_PROVIDER=mock, email was not actually sent.",
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
        base = {"to": to, "subject": subject, "body": body, "body_preview": body[:200]}
        if not self.host or not self.username:
            return {
                **base,
                "success": False,
                "provider": "smtp",
                "sent": False,
                "message": "SMTP not configured. Set SMTP_HOST and SMTP_USERNAME.",
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
                **base,
                "success": True,
                "provider": "smtp",
                "sent": True,
                "message": "Email sent successfully.",
            }
        except Exception as exc:
            logger.error("SMTP send failed: %s", exc)
            return {
                **base,
                "success": False,
                "provider": "smtp",
                "sent": False,
                "message": f"SMTP send failed: {exc}",
            }


def get_email_provider() -> MockEmailProvider | SMTPEmailProvider:
    provider = os.getenv("EMAIL_PROVIDER", "mock").lower()
    if provider == "smtp":
        return SMTPEmailProvider()
    return MockEmailProvider()
