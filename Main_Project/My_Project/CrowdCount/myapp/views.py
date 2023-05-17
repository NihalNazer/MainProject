from django.shortcuts import render, redirect
from myapp.models import UploadedFile

def upload(request):
    if request.method == 'POST':
        file = request.FILES['file-input']
        uploaded_file = UploadedFile(file=file)
        uploaded_file.save()
        return redirect('upload')  # Redirect to the same page after successful upload
    return render(request, 'upload.html')
