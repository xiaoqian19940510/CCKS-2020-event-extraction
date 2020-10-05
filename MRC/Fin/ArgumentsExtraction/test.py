from Io.data_loader import create_batch_iter
from preprocessing.data_processor import read_squad_data, convert_examples_to_features, read_qa_examples
from pytorch_pretrained_bert.tokenization import BertTokenizer
from predict.predict import main

if __name__ == "__main__":
    read_squad_data("/home/LAB/liqian/test/game/Fin/CCKS-Mrc/data/squad_like_test.json", "/home/LAB/liqian/test/game/Fin/CCKS-Mrc/data/",is_training=False)
    # examples = read_qa_examples("/home/LAB/liqian/test/game/ccks-2020-finance-transfer-ee-baseline-master/CCKS-Mrc/data/", "test")
    examples = read_qa_examples("/home/LAB/liqian/test/game/Fin/CCKS-Mrc/data/", "test")
    main('/home/LAB/liqian/test/game/Fin/CCKS-Mrc/data/')
