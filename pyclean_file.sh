FILE=$1
PROJECT="/home/derek/Desktop/lactate"
PYTHON="$HOME/.pyenv/versions/3.8.5/bin/python3.8" # change this to the correct python for you
SRC="/home/derek/Desktop/lactate/lactate"
TST="/home/derek/Desktop/lactate/test"

# eval cd $PROJECT
echo "Fixing $FILE with autoflake..." && $PYTHON -m autoflake --in-place --remove-unused-variables --remove-all-unused-imports -r $FILE
echo "Formatting $FILE imports with isort..." && $PYTHON -m isort $FILE
echo "Formatting $FILE with black..." && $PYTHON -m black $FILE
