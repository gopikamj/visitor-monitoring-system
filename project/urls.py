from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from app import views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", views.index, name='index'),
    path("register/", views.visitor_register, name='register'),
    path("login/", views.visitor_login, name='login'),
    path("logout/", views.visitor_logout, name='logout'),
    path("add_visitor/", views.add_visitor, name='add_visitor'),
    path("visitors/", views.visitor_list, name='visitor_list'),
    path("blocked_visitors/", views.blocked_visitor_list, name='blocked_visitor_list'),
    path("block_visitor/<int:visitor_id>/", views.block_visitor, name='block_visitor'),
    path("unblock_visitor/<int:visitor_id>/", views.unblock_visitor, name='unblock_visitor'),
    path("livemonitering/", views.livemonitering, name='livemonitoring'),
    path("livemonitering/", views.livemonitering, name='livemonitoring'),
    path("allowuser/", views.allowuser, name='allowuser'),
    path("Resetuser",views.Resetuser),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
