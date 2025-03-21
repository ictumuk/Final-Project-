from django.contrib.sessions.middleware import SessionMiddleware
from django.conf import settings

class AdminSessionMiddleware(SessionMiddleware):
    def process_request(self, request):
        if request.path.startswith('/admin/'):
            request.session = self.SessionStore(
                request.COOKIES.get(settings.ADMIN_SESSION_COOKIE_NAME, None)
            )
        else:
            super().process_request(request)

    def process_response(self, request, response):
        if request.path.startswith('/admin/'):
            if request.session.modified:
                request.session.save()
            session_cookie = request.session.session_key
            response.set_cookie(
                settings.ADMIN_SESSION_COOKIE_NAME,
                session_cookie,
                max_age=settings.SESSION_COOKIE_AGE,
                path=settings.ADMIN_SESSION_COOKIE_PATH,
                domain=settings.SESSION_COOKIE_DOMAIN,
                secure=settings.SESSION_COOKIE_SECURE or None,
                httponly=settings.SESSION_COOKIE_HTTPONLY or None,
                samesite=settings.SESSION_COOKIE_SAMESITE,
            )
            return response
        return super().process_response(request, response)