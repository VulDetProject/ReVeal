echo "Downloading models"
FILE=models.zip
if [[ -f "$FILE" ]]; then
    echo "$FILE exists, skipping download"
else
    # https://drive.google.com/file/d/1gTgpgXGzSBlixNcUS-OaoXe8HxQXzaf0
    fileid="1gTgpgXGzSBlixNcUS-OaoXe8HxQXzaf0"
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${FILE}
    rm ./cookie
    unzip ${FILE} && rm ${FILE}
fi
