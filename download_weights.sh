filename=model.pth
if test -f "$filename";
then
    echo "$filename has found."
else
    echo "$filename has not been found...downloading"
    aws s3 cp s3://servingapi/model.pth model.pth --no-sign-request
fi