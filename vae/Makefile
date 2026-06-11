ARGS=-I../lib -Wall -Wextra ../lib/util.c fig.c vae.c mnist.c -lOpenCL -lm -Wno-unused-result

all:
	@echo "interp   - create interpolation between two images"
	@echo "csv      - write latent space to csv"
	@echo "traverse - traverse across latent dimensions"
	@echo "heatmap  - create heatmap of output wrt latent dimensions"
	@echo "train    - train a model"

interp: bin
	@mkdir -p interp
	@mkdir -p interp/vae
	@mkdir -p interp/ae
	gcc $(ARGS) -Og -g3 interp.c -o bin/interp.exe

csv: bin
	@mkdir -p csv
	gcc $(ARGS) -O2 csv.c -o bin/csv.exe 

traverse: bin
	@mkdir -p traverse
	@mkdir -p traverse/vae
	@mkdir -p traverse/ae
	gcc $(ARGS) -O2 traverse.c -o bin/traverse.exe

heatmap: bin
	@mkdir -p heatmap
	gcc $(ARGS) -O2 heatmap.c -o bin/heatmap.exe

train: bin
	@mkdir -p images
	gcc $(ARGS) -g3 train.c -o bin/train.exe

bin:
	mkdir -p bin

.PHONY: interp csv debug clean traverse heatmap train
