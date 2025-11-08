
filename_brg=galaxy_general
if [ -f "${filename_brg}.g1" ]; then
    echo -e "There exist ${filename_brg}.g1. Continue."
else
    echo -e "There not exist ${filename_brg}.g1. Exit."
    exit
fi

filename_snapshot_choose=snapshot_020
cd ../template_run/
echo -e "$PWD"
bash 0_Run.bat
cd ../step1_from_ascii_to_g1_and_run/
echo -e "$PWD"

cp ../template_run/snapshot/${filename_snapshot_choose} ${filename_brg}.g1
./read_snapshot.exe 2 ${filename_brg}.g1
mv ${filename_brg}.g1.txt ${filename_brg}.txt
