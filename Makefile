install:
	python setup.py build_ext --inplace

clean:
	rm -f ./*/*.c
	rm -f ./*/*.html
	rm -f ./*/*.so
	rm -rf ./build/
	rm -rf __pycache__/
	rm -rf ./*/__pycache__/
	rm -rf ./decoding/decoding_*.cpp
