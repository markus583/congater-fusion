# go thoruhg all sub-directories recursively and delete all directories starting with "checkpoint-" recursively
find . -type d -name 'checkpoint-*' -exec rm -rf {} +