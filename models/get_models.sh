echo "Downloading models"
FILE=models.zip
if [[ -f "$FILE" ]]; then
    echo "$FILE exists, skipping download"
else
    # https://drive.google.com/open?id=1plKRZ_tJZJQR7REr0QnTbBOw49Rh2ZNQ
    fileid="1plKRZ_tJZJQR7REr0QnTbBOw49Rh2ZNQ"
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${FILE}
    rm ./cookie
    unzip ${FILE} && rm ${FILE}
fi
