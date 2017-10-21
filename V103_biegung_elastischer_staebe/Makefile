all: build/rundeinseitig.pdf build/quadratischeinseitig.pdf build/quadratischzweiseitig.pdf

build/rundeinseitig.pdf: auswertung.py rundeinseitig.txt | build
	python auswertung.py

build/quadratischeinseitig.pdf: auswertung.py quadratischeinseitig.txt | build
	python auswertung.py

build/quadratischzweiseitig.pdf: auswertung.py quadratischzweiseitig.txt | build

build:
	mkdir -p build

clean:
	rm -rf build
