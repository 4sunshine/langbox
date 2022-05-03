# Language + Vision pipelines playground
Implementation of own ideas during my free time. Mainly in area of NLP and CV.  
Written only for educational and research purposes.  

## Contents
> 1. Fake news generation pipeline.

## Fake news generation
This solution allows to train and do inference of **ru-gpt3** model on Telegram channel data.  
After that you can run **ru-dalle** model in low-resource friendly mode on generated text.
### Pipeline
1. Download any Telegram channel data in `.json` format. You can use the desktop version of messenger for this.  
2. Parse downloaded data with script in training-ready format: 
```
python utils/parse_telegram.py --data_path /PATH/TO/DOWNLOADED/JSON.json --start_date 1980-01-01T00:00:00 --special_remove STRINGS;OF;WORDS;TO;REMOVE
```
3. Optionally you can do own further data cleaning.
4. Run to randomly split on **train / val** sets:
```
python dataset/utils.py /PATH/TO/PARSED/DATA.txt PORTION_OF_VAL
```
5. Train ru-gpt3 model on prepared data:
```
python train_telegram.py --model_type /PATH/TO/rugpt3small_based_on_gpt2 --input_file /PATH/TO/train.txt --val_file /PATH/TO/val.txt --run_name EXPERIMENT_NAME --save_every 180 --n_epochs 10 --sample_file /PATH/TO/sample.txt --lr 0.00005
```
Here `sample.txt` is a file with the beginnings of sentences you want to be generated. For example:
```
Британские учёные открыли
Российские физики из ННГУ им. Н.И. Лобачевского
Астрономы обнаружили
```
Sampling will occur at every validation step. You can use own schedulers, optimizers & metrics.
LM loss used for best model selection.
6. Run trained language model:  
```
python run_gpt.py /PATH/TO/BEST_OR_FINAL/CHECKPOINT /PATH/TO/sample.txt /PATH/TO/rut5_paraphrase
```
Where `sample.txt` is the beginnings file in format from previous step.
This script also performs keyword extraction from generated text to use it as input to DALL-e model.
I use paraphraser to make extracted texts more natural.  
As a result there will be two files generated: `predict_sample.txt` with "fake news" and `generation_predict_sample.txt` with extracted keywords.  
#### Examples: 
predict_sample.txt:  
```
Британские учёные открыли новый тип графов — латинские символы с крутыми вершинами. Это стало возможным благодаря исследованию числа Δ (или единицы — это не «знак умножения», а просто число) и, как следствие, произошло изменение знака препинания в древнем и современном языках.
Российские физики из ННГУ им. Н.И. Лобачевского создали квантовый компьютер, в котором реализована технология двойных преобразований информации. А теперь можно проверить работу системы на сжатие.
Астрономы обнаружили на границе созвездия Венера протопланету, которая будет наблюдать рождение в нем нестабильных звезд. По расчетам ученых, они могут быть связаны с ростом масс звезд, расположенными близко к Земле.
```

generation_predict_sample.txt:
```
новый тип графов и изменение знака препинания
двойные преобразования информации и российские физика
рост масс звёзд и расчёты учёных
```
7. Run rudalle model on generated texts:  
```
python run_rudalle.py /PATH/TO/generation_predict_sample.txt --rudalle_path /PATH/TO/rudalle_malevich --rudalle_name Malevich
```
This script will do inference of rudalle model, sort generated pictures with ru-clip model, apply x2 resolution & store 
top-clip-score generated pictures in folder `dalle_generation_predict_sample`. Inference tested on cuda-11.4 and
NVIDIA RTX2070 Super with 8 Gb VRAM.  

### References
Project code mainly based on:  
- [Meet your Artificial Self: Generate text that sounds like you](https://github.com/mar-muel/artificial-self-AMLD-2020)  
- [Malevich-3.5GB-vRAM-usage.ipynb](https://colab.research.google.com/drive/1AoolDYePUpPkRCKIu0cP9zV7lX5QGD3Z)
- [Russian GPT-3 models](https://github.com/ai-forever/ru-gpts)
- [ruDALL-E](https://github.com/ai-forever/ru-dalle)
- [cointegrated/rut5-base-paraphraser](https://huggingface.co/cointegrated/rut5-base-paraphraser)
- [rutermextract](https://github.com/igor-shevchenko/rutermextract)

GPT-3 pretrained model:
- [sberbank-ai/rugpt3small_based_on_gpt2](https://huggingface.co/sberbank-ai/rugpt3small_based_on_gpt2)
