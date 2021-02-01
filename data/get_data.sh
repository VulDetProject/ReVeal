echo "Downloading Replication dataset"
FILE=replication.zip
if [[ -f "$FILE" ]]; then
    echo "$FILE exists, skipping download"
else
    # https://drive.google.com/file/d/1Mn0jLaZWiPFQ8ejzlz_zXnx_TcSzbwu1
    fileid="1Mn0jLaZWiPFQ8ejzlz_zXnx_TcSzbwu1"
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${FILE}
    rm ./cookie
    unzip ${FILE} && rm ${FILE}
fi
