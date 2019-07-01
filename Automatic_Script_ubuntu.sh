#!/bin/sh
sudo apt-get update
sudo apt-get install python3.6 -y
sudo apt-get -y install python3-pip
sudo apt install git
git clone https://github.com/vt1998/Save_Electricity_using_ML
cd Save_Electricity_using_ML
pip3 install -r requriment.txt
sudo apt-get install mysql-server -y
sudo mysql <<MYSQL_SCRIPT
CREATE DATABASE pythonlogin;
USE pythonlogin;

CREATE TABLE accounts (
        id int(11) NOT NULL AUTO_INCREMENT,
        username varchar(50) NOT NULL,
        password varchar(255) NOT NULL,
        email varchar(100) NOT NULL,
        mobile varchar(20) NOT NULL,
    PRIMARY KEY (id)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8;

INSERT INTO accounts (id, username, password, email) VALUES (1, 'test', 'test', 'test@test.com', '1234567890');
MYSQL_SCRIPT

echo "Database created you are ready to go"
echo "go to Save_Electricity_using_ML"
echo "RUN COMMANDS of FLASK"
echo "set FLASK_APP=main.py"
echo "FLASK run"