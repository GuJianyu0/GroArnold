
local_is_trans_txt=${1}
filename_brg=galaxy_general
if [ ${local_is_trans_txt} -eq 0 ]; then
    echo -e "from g1 to txt"
    ./read_snapshot.exe 2 ${filename_brg}.g1 #to get Input.txt and Input.txt.SCF from Input
    mv ${filename_brg}.g1.txt ${filename_brg}.txt
else
    echo -e "from txt to g1"
    ./read_snapshot.exe 1 galaxy_general.txt #to get Input.g1 and Input.g1.SCF from Input
    mv galaxy_general.txt.g1 galaxy_general.g1
fi
