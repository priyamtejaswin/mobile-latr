Move all data here.

Uses arrow files for faster reads and zero-copying.

# Fine-tuning QA

Every entry in the arrow table should have the following columns:
```bash
{'image_id': '003a8ae2ef43b901',
 'question_id': 34602,
 'question': 'what is the brand of this camera?',
 'question_tokens': ['what', 'is', 'the', 'brand', 'of', 'this', 'camera'],
 'image': <byte array of image>,
 'image_width': 1024,
 'image_height': 664,
 'flickr_original_url': 'https://farm2.staticflickr.com/4/5566811_bc00d504a6_o.jpg',
 'flickr_300k_url': 'https://farm2.staticflickr.com/4/5566811_bc00d504a6_o.jpg',
 'answers': ['nous les gosses',
  'dakota',
  'clos culombu',
  'dakota digital',
  'dakota',
  'dakota',
  'dakota digital',
  'dakota digital',
  'dakota',
  'dakota'],
 'image_classes': ['Cassette deck',
  'Printer',
  'Medical equipment',
  'Computer mouse',
  'Scale',
  'Telephone',
  'Camera',
  'Ipod',
  'Remote control'],
 'set_name': 'val'}
```
