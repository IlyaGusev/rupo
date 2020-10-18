wget https://www.dropbox.com/s/dwkui2xqivzsyw5/generator_model.zip
mkdir -p ./rupo/data/generator_models
unzip generator_model.zip -d ./rupo/data/generator_models
rm generator_model.zip

wget https://www.dropbox.com/s/ajd8b7lpqaao7xt/stress_ru_main.tar.gz
mkdir -p ./rupo/data/stress_models/ru_main
tar -xzvf stress_ru_main.tar.gz --directory ./rupo/data/stress_models/ru_main
rm stress_ru_main.tar.gz

wget https://www.dropbox.com/s/7rk135fzd3i8kfw/g2p_models.zip
mkdir -p ./rupo/data/g2p_models
unzip g2p_models.zip -d ./rupo/data/g2p_models
rm g2p_models.zip

wget https://www.dropbox.com/s/znqlrb1xblh3amo/dict.zip
mkdir -p ./rupo/data/dict
unzip dict.zip -d ./rupo/data/dict
rm dict.zip
