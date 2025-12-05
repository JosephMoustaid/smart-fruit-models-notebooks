# Smart Fruit Recognition System

A computer vision project that uses deep learning to classify fruits by type, ripeness, and diseases. Built with TensorFlow/Keras and optimized for mobile deployment with TensorFlow Lite.

## What This Does

I needed a way to automatically identify fruits and check their quality, so I built three models that work together:

1. **Fruit Type Recognition** - Tells you what fruit you're looking at (apple, banana, orange, etc.)
2. **Ripeness Detection** - Figures out if the fruit is fresh, ripe, or rotten
3. **Disease Classification** - Spots common fruit diseases like blotch, rot, or anthracnose

The models are small enough to run on a phone, which is the whole point - you can take a photo and get instant results.

## The Models

All three use **MobileNetV2** as the base architecture because it's fast and actually works on mobile devices. I'm using transfer learning with ImageNet weights, then training custom classification heads on top.

### Training Setup

- **Image size**: 96×96 (optimal for MobileNetV2)
- **Batch size**: 128 (for faster training)
- **Epochs**: 12 (enough to get good results without overfitting)
- **Optimizer**: Adam with 0.0001 learning rate
- **Data augmentation**: rotation, shifts, zoom, horizontal flips

Each model trains in about 5-10 minutes on CPU, which is way better than the hours it was taking before optimization.

## Dataset Info

Using three different datasets:

- **Fruits-360**: 100+ fruit/vegetable types with 100 images per class for training
- **Fruit Ripeness Dataset**: Fresh, ripe, and rotten examples for apples, bananas, and oranges  
- **Fruit Disease Dataset**: Apple, guava, mango, and pomegranate diseases (real agricultural data)

The datasets aren't included in this repo because they're huge. You'll need to download them separately if you want to retrain the models.

## Files You'll Find Here

```
smart-fruit/
├── Model 1 - Fruits and veggies recognition.ipynb
├── model 2 - Fruits ripe-unripe image classficaiton.ipynb
├── Model 3 - Fruits disease classification.ipynb
├── *.h5                    # Trained Keras models
├── *.tflite               # Converted TFLite models (for mobile)
├── *_labels.txt           # Class names in order
└── README.md
```

## Running the Notebooks

Just open any notebook and run the cells in order. They're set up to:
1. Load the local datasets (make sure paths are correct)
2. Build and train the model
3. Save the trained model (.h5 format)
4. Convert to TensorFlow Lite (.tflite)
5. Export the class labels (.txt)

The notebooks have comments explaining what's happening at each step.

## Why TensorFlow Lite?

Regular TensorFlow models are too big for mobile apps. TFLite uses quantization to shrink the models down - we're talking from ~15MB to ~4MB - while keeping accuracy high. The converted models are ready to drop into a Flutter app or any mobile framework that supports TFLite.

## Performance Notes

Training on CPU isn't ideal but it works. If you have a GPU set up with CUDA, it'll use it automatically and train way faster. The current setup is optimized for speed even without GPU:

- Small image sizes (96×96)
- Large batch sizes (128)
- Efficient data loading
- No unnecessary augmentation on validation data

## What's Next

The models work pretty well but there's always room for improvement:
- Fine-tuning by unfreezing some base model layers
- Testing different architectures (EfficientNet, etc.)
- More epochs if you have time
- Collecting more training data for edge cases

## Technical Stack

- Python 3.12
- TensorFlow/Keras 2.x
- NumPy, Matplotlib
- scikit-learn (for dataset splitting in Model 3)

---

Built this because I got tired of guessing which fruits at the market were actually good. Now I just point my phone at them.
