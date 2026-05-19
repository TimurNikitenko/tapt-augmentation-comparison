import re
import emoji


class TextProcessor:
    """Предобработка текста для RuBERT"""

    def __init__(self):
        self.html_pattern = re.compile(r"<[^>]+>")

        self.video_url_pattern = re.compile(
            r"https?://(?:www\.)?(?:youtube\.com|youtu\.be|vk\.com/video|rutube\.ru|vimeo\.com|music\.yandex\.ru|podcasts\.apple\.com)[^\s]+"
        )
        self.url_pattern = re.compile(r"https?://[^\s]+")

        self.mail_pattern = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
        self.phone_pattern = re.compile(r"[\+]?[0-9\s\-\(\)]{10,}")
        self.multiple_spaces = re.compile(r"\s+")

    def preprocess(self, text: str) -> str:
        """Предобработка текста для классификации"""
        if not isinstance(text, str) or len(text) == 0:
            return "empty"

        

        text = text.lower()
        text = emoji.demojize(text, language="ru")
        
        text = self.html_pattern.sub(" ", text)
        text = self.video_url_pattern.sub(" [MEDIA_URL] ", text)
        text = self.url_pattern.sub(" [URL] ", text)
        text = self.mail_pattern.sub(" [EMAIL] ", text)
        text = self.phone_pattern.sub(" [PHONE] ", text)

        text = self.multiple_spaces.sub(" ", text)

        text = text.strip()

        if len(text) <= 10:
            return "empty"

        words = text.split()

        if len(words) <= 400:
            return text

        return " ".join(words[:400])
