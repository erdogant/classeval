echo "Cleaning.."
make clean

echo "Create html.."
make html

echo "Make pdf.."
sphinx-build -b pdf source build
