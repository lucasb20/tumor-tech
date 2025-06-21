* Etapas seguidas:

- Para a seção do Pré-processamento(3.3), peguei todas as imagens da base de dados, disponível em https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri
- Depois, usando o código do arquivo 'grayscale.py', coloquei todas as imagens em escala de cinza e tamanho 224x224;
- A seguir, usando o código do arquivo 'bilateral-filtering.py', apliquei filtragem bilateral para suavizar as imagens preservando as bordas;
- Depois disso, usando o código do arquivo 'clahe.py', é aplicada equalização de histograma adaptativo limitado por contraste (CLAHE) a todas as imagens, para aumentar a nitidez;
- Por último, usando o código do arquivo 'augmented_date.py', foi feito o aumento de dados, criando várias diferentes imagens para cada imagem processada até o item anterior, apenas do tipo Training (SEÇÃO 3.4).

* O que pode ser alterado em cada programa para tentar melhorar as imagens em relação aos parâmetros:

- No programa 'bilateral-filtering.py':

´´´
imgFilter = cv.bilateralFilter(img, 7, 30, 5)
´´´

Altere os valores 7, 30, 5 para obter diferentes resultados. O primeiro número é o diâmetro, que quanto maior é feito mais suavização. O segundo número é a diferença de intensidade (contraste), que quanto maior suaviza regiões com diferenças maiores de cor. O último número é a distância entre pixels, que quanto maior aumenta o alcance espacial do filtro.

- No programa 'clahe.py':

´´´
clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)) 
´´´

Altere os valores de clipLimit e titleGridSize para obter diferentes resultados. clipLimit controla o quanto de contraste pode ser aumentado, a faixa entre 2.0 e 4.0 costuma ser boa. titleGridSize define o tamanho dos blocos locais.

* Importante:

Para executar cada código, é necessário alterar os caminhos das pastas de entrada (imagens que serão processadas) e de sáida (local de destino das imagens processadas), por exemplo:

No código 'bilateral-filtering.py', eu chamei a pasta de entrada no caminho 'Imagens/Testing_processed/glioma_tumor' para aplicar filtro bilateral as imagens dela e mandar os resultados para a pasta de saída do caminho 'Imagens/Testing_filtered/glioma_tumor', ou seja, para cada tipo de tumor cerebral, é necessário alterar a parte antes de '_tumor' para executar o código em cima da pasta desejada. Isso vale também para os códigos 'clahe.py' e 'augmented_date.py'.
