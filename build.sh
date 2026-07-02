echo   '构建 wheel 包'
python -m build --wheel


echo "构建完成，开始上传 wheel 包到 PyPI"
twine upload dist/*