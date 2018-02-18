# Find nearest image

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/Donskov7/find_nearest_image/blob/master/LICENSE)

Web service on Django to find nearest image to the uploaded one in database

[![Example](https://img.youtube.com/vi/WrgNBYMUhQw/0.jpg)](https://www.youtube.com/watch?v=WrgNBYMUhQw)

# Dataset
I used [mirflickr](http://press.liacs.nl/mirflickr/#sec_download) dataset as image corpus for database.
It should be uploaded to `media/documents` folder.

# Requirements
- numpy==1.14.0
- scipy==1.0.0
- h5py==2.7.1
- sklearn==0.19.0
- tensorflow==1.3.0
- keras==2.1.3
- django==2.0.2

Library is compatible with: __Python 2.7-3.6__.

# How to use
1. Fill database with images

```python put_data_to_db.py --db db.sqlite3 --image-path media/documents/mirflickr --drop-prev-db --output-dir . --table myapp_images```

2. Run service

```python manage.py runserver```

3. Go to ```http://127.0.0.1:8000/``` 

4. Try upload some image

# How does it work
Database filled with **mirflickr** images. 
Each row containes `[filename, hash, image_vector, image_vector_pca]`:
  - `filename`: path to image
  - `hash`: hash from image vector
  - `image_vector`: vector of floats describing the image. It's the "fc2" layer output from VGG16.
  - `image_vector_pca`: `image_vector` after PCA.

Every uploaded image go through VGG16: gets `image_vector`.
Then `image_vector` go through PCA.
Then I use KDTree search (over `image_vector_pca`) to find 5 candidates from database.
The nearest image would have the minimal cosine distance with uploaded `image_vector`.

# What could be improved
1. VGG16 is't the best neural network to get image description vector.
2. Also it's necessary to avoid global variables.
