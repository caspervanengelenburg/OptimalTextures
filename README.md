# OptimalTextures
An implementation of [Optimal Textures: Fast and Robust Texture Synthesis and Style Transfer through Optimal Transport](https://arxiv.org/abs/2010.14702) for TU Delft CS4240.

This repository is a WIP.

[Paper notes](notes.md)

```bash
git clone https://github.com/JCBrouwer/OptimalTextures
cd OptimalTextures
pip install -r requirements.txt
python optex.py -h

# texture synthesis
python optex.py -s style/graffiti.jpg --size 512

# style transfer
python optex.py -s style/lava-small.jpg -c content/rocket.jpg --content_strength 0.2

# texture mixing
python optex.py -s style/zebra.jpg style/pattern-small.jpg --mixing_alpha 0.5  

# color transfer
python optex.py -s style/green-paint-large.jpg -c content/city.jpg --style_scale 0.5 --content_strength 0.2 --color_transfer opt --size 1024
```
