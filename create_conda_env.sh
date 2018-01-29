while true; do
    read -p "Do you wish to create a new virtual env named 'ucsd' and install all of its dependencies?" yn
    case $yn in
        [Yy]* ) conda env create -f environment.yml; break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done
