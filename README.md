# Sketch-Generation-with-Drawing-Process-Guided-by-Vector-Flow-and-Grayscale
This is the official implementation of the **AAAI 2021** accepted paper "**Sketch Generation with Drawing Process Guided by Vector Flow and Grayscale**". Our Supplementary Material (PDF) is available at Baidu Netdisk (百度网盘) https://pan.baidu.com/s/1ZZEmpuPXRNBPvG0WHbbPKA. The extraction code (提取码) is `1234`.
Here we give some instructions for running our code.
## Authors
### First author
Zhengyan Tong (童峥岩)， 此论文idea提供者以及代码原作者，主要从事计算机视觉方面的研究。发表此论文时为 上海交通大学 电子信息与电气工程学院 信息工程专业 大四在读本科生。
### Other author
Xuanhong Chen (Shanghai Jiao Tong University Ph. D.)
Bingbing Ni (Shanghai Jiao Tong University Associate Professor)
Xiaohang Wang (Shanghai Jiao Tong University undergraduate)

## Examples
We give three examples that can be run directly (the hyperparameters of these three examples have been fixed).  
### Quick start
- To draw the cat: `python cat.py`
- To draw the dog: `python dog.py`
- To draw the girl: `python girl.py`
### Results
<div style="white-space: nowrap;">
<img src="https://raw.githubusercontent.com/TZYSJTU/Sketch-Generation-with-Drawing-Process-Guided-by-Vector-Flow-and-Grayscale/master/Supplementary-Material/cat.gif" height="285" alt="cat"/> <img src="https://raw.githubusercontent.com/TZYSJTU/Sketch-Generation-with-Drawing-Process-Guided-by-Vector-Flow-and-Grayscale/master/Supplementary-Material/dog.gif" height="285" alt="cat"/> <img src="https://raw.githubusercontent.com/TZYSJTU/Sketch-Generation-with-Drawing-Process-Guided-by-Vector-Flow-and-Grayscale/master/Supplementary-Material/girl.gif" height="285" alt="cat"/>
</div>

<div style="white-space: nowrap;">
<img src="https://raw.githubusercontent.com/TZYSJTU/Sketch-Generation-with-Drawing-Process-Guided-by-Vector-Flow-and-Grayscale/master/Supplementary-Material/cat_RGB.jpg" height="285" alt="cat"/> <img src="https://raw.githubusercontent.com/TZYSJTU/Sketch-Generation-with-Drawing-Process-Guided-by-Vector-Flow-and-Grayscale/master/Supplementary-Material/dog_RGB.jpg" height="285" alt="cat"/> <img src="https://raw.githubusercontent.com/TZYSJTU/Sketch-Generation-with-Drawing-Process-Guided-by-Vector-Flow-and-Grayscale/master/Supplementary-Material/girl_RGB.jpg" height="285" alt="cat"/>
</div>

## Instructions
To draw arbitrary input: `python process_order.py`. Of course you need to adjust the following parameters.
### Hyperparameters
- `input_path = './input/your file'` Input image path
- `output_path = './output'` Do not change this
- `n =  10` Gray-scale quantization order
- `period = 5` Line(stroke) width
- `direction =  10` Direction quantization order
- `Freq = 100` Save the drawing process every `Freq` strokes are drawn
- `deepen =  1` Edge map's intensity. The bigger, the darker.
- `transTone = False` Do not change this
- `kernel_radius = 3` Edge tangent flow kernel size, do not change this
- `iter_time = 15` Edge tangent flow kernel iterations times, do not change this
- `background_dir = None`  Whether fix the drawing direction in the background, this value could be `None` or an integer between `(0~180)`
- `CLAHE = True` Whether input uses CLAHE (Do not change this)
- `edge_CLAHE = True` Whether edge map uses CLAHE (Do not change this)
- `draw_new = True` Do not change this
- `random_order = False` Use random drawing order if `True`
- `ETF_order = True` Use the drawing order described in our paper if `True`
- `process_visible = True` Whether show the drawing process 

In our supplementary material (PDF), we explain these hyperparameters in more detail and we show more comparisons with existing pencil drawing algorithms. We also offer more
results of our method. Our Supplementary Material is available at Baidu Netdisk (百度网盘) https://pan.baidu.com/s/1ZZEmpuPXRNBPvG0WHbbPKA. The extraction code (提取码) is `1234`.
### Results
<div align=center style="white-space: nowrap;">
<img src="https://raw.githubusercontent.com/TZYSJTU/Sketch-Generation-with-Drawing-Process-Guided-by-Vector-Flow-and-Grayscale/master/Supplementary-Material/girl2.gif"   alt="girl2"/> 
</div>

<div align=center style="white-space: nowrap;">
<img src="https://raw.githubusercontent.com/TZYSJTU/Sketch-Generation-with-Drawing-Process-Guided-by-Vector-Flow-and-Grayscale/master/Supplementary-Material/girl2-RGB.png"   alt="girl2"/> 
</div>

<div align=center style="white-space: nowrap;">
<img src="https://raw.githubusercontent.com/TZYSJTU/Sketch-Generation-with-Drawing-Process-Guided-by-Vector-Flow-and-Grayscale/master/Supplementary-Material/beauty.gif"   alt="girl2"/> 
</div>

<div align=center style="white-space: nowrap;">
<img src="https://raw.githubusercontent.com/TZYSJTU/Sketch-Generation-with-Drawing-Process-Guided-by-Vector-Flow-and-Grayscale/master/Supplementary-Material/beauty-RGB.jpg"   alt="girl2"/> 
</div>
