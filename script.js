import { AutoTokenizer, AutoModelForSequenceClassification} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2';

const modelName = "arajun/ruri-v3-70m-wrime-onnx";

class Classificator{

    static async loadModel(modelName) {
        const classificator = new Classificator();
        classificator.tokenizer = await AutoTokenizer.from_pretrained(modelName);
        classificator.model = await AutoModelForSequenceClassification.from_pretrained(modelName, {device: 'auto', dtype: 'fp32'});
        return classificator;
    }

    async classify(input_text) {
        const input = await this.tokenizer(input_text);
        const output = await this.model(input);

        function softmax(logits) {
            const maxLogit = Math.max(...logits);
            const exps = logits.map(x => Math.exp(x - maxLogit)); // オーバーフロー防止
            const sumExps = exps.reduce((a, b) => a + b, 0);
            return exps.map(x => x / sumExps);
        }
        function sigmoid(logits) {
            return logits.map(x => 1 / (1 + Math.exp(-x)));
        }

        return sigmoid(output.logits.data);
    }
}

let classificator;

const button = document.getElementById("classify");

const loadModel = async()=>{
    button.innerText = "モデル読込中";
    button.disabled = true;

    classificator = await Classificator.loadModel(modelName)

    button.innerText = "感情分析だ！！！";
    button.disabled = false;
    button.onclick = classify;
}

button.onclick = loadModel;

const classify = async()=>{
    const text = document.getElementById("text").value;

    const output = await classificator.classify(text)

    document.getElementById("result").innerHTML = "";

    output.forEach((x,i)=>{
        const tr = document.createElement("tr");
        const td1 = document.createElement("td");
        const td2 = document.createElement("td");

        td1.innerText = classificator.model.config.id2label[i];
        td2.innerText = Math.round(x*1000)/10;
        tr.appendChild(td1);
        tr.appendChild(td2);
        
        document.getElementById("result").appendChild(tr);
    });
}