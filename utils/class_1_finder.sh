while read f;
do
    [ -f "$f".mhd ] && echo "$f" exists && cp "$f".* ./subsetME
done < CT_scans_class_1.txt
