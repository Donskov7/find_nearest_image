import os.path
from django.shortcuts import render
from django.template import RequestContext
from django.http import HttpResponseRedirect
from django.urls import reverse

import tensorflow as tf

from .models import Document
from .forms import DocumentForm
from .search import search_nearest

from zedge.settings import MEDIA_ROOT, DATABASES
from imglib.utils import VGG

MODEL = VGG()
global GRAPH
GRAPH = tf.get_default_graph()

def list(request):
    # Handle file upload
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            newdoc = Document(docfile=request.FILES['docfile'])
            newdoc.save()
            fname = os.path.join(MEDIA_ROOT, str(newdoc.docfile))
            with GRAPH.as_default():
                nearest_doc = search_nearest(DATABASES['default']['NAME'], 'myapp_images', fname, MODEL)
                if nearest_doc is not None:
                    data = {
                            'document0': '../../media/{}'.format(newdoc.docfile),
                            'document1': '../../media/documents/mirflickr/{}'.format(nearest_doc['file'].split('/')[-1]),
                            'form': form
                    }
                    return render(request, 'list.html', data)
    else:
        form = DocumentForm()  # A empty, unbound form


    # Render list page with the documents and the form
    return render(request, 'list.html', {'form': form})
