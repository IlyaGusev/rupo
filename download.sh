wget https://www.dropbox.com/s/eefgbo53e000by5/generator_models.zip
mkdir -p ./rupo/data/generator_models
unzip generator_models.zip -d ./rupo/data/generator_models
rm generator_models.zip

wget https://www.dropbox.com/s/31pyubqskp4krsd/stress_models_light.zip
mkdir -p ./rupo/data/stress_models
unzip stress_models_light.zip -d ./rupo/data/stress_models
rm stress_models.zip

wget https://www.dropbox.com/s/7rk135fzd3i8kfw/g2p_models.zip
mkdir -p ./rupo/data/g2p_models
unzip g2p_models.zip -d ./rupo/data/g2p_models
rm g2p_models.zip

wget https://www.dropbox.com/s/znqlrb1xblh3amo/dict.zip
mkdir -p ./rupo/data/dict
unzip dict.zip -d ./rupo/data/dict
rm dict.zip