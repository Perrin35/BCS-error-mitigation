OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg c[3];
rz(4.49024737663343) q[18];
sx q[18];
rz(4.78203980636384) q[18];
sx q[18];
rz(11.5358847245921) q[18];
rz(4.17968256578973) q[21];
sx q[21];
rz(4.20345982577717) q[21];
sx q[21];
rz(10.7303293883543) q[21];
rz(3.06830634039562) q[23];
sx q[23];
rz(3.50182057435469) q[23];
sx q[23];
rz(14.4142147547712) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[18];
rz(-pi) q[21];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
rz(-pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
x q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[21];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[18];
x q[18];
rz(-pi) q[23];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
rz(-pi) q[21];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[18];
rz(-pi) q[23];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[18];
rz(-pi) q[21];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
rz(-pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[18];
x q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[21];
rz(pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
x q[21];
rz(-pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[18];
x q[21];
rz(pi) q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
rz(-pi) q[23];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
rz(-pi) q[23];
x q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
x q[18];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
rz(-pi) q[21];
x q[21];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[18];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[18];
rz(-pi) q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
x q[21];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
x q[21];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[18];
x q[18];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
x q[21];
x q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
x q[21];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[21];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[18];
rz(-pi) q[21];
rz(-pi) q[23];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[18];
rz(-pi) q[21];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[18];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[21];
x q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
rz(-pi) q[23];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[23];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[18];
x q[18];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
x q[21];
x q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
rz(-pi) q[21];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[18];
rz(-pi) q[21];
x q[21];
rz(-pi) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[18];
rz(-pi) q[21];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
rz(-pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[18];
x q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
x q[21];
rz(-pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[18];
x q[18];
rz(-pi) q[21];
x q[21];
rz(pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
x q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[18];
x q[21];
rz(-pi) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[21];
x q[21];
rz(pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
rz(-pi) q[23];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
x q[21];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[18];
rz(pi) q[21];
rz(-pi) q[23];
x q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
rz(-pi) q[23];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
x q[21];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
x q[18];
rz(-pi) q[21];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[21];
rz(-pi) q[23];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[18];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
rz(-pi) q[21];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
rz(-pi) q[21];
x q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[21];
x q[21];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
x q[21];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[18];
x q[18];
rz(pi) q[21];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
x q[21];
rz(-pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[18];
rz(-pi) q[21];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[21];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
x q[18];
rz(-pi) q[21];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[18];
x q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
x q[21];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
x q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[18];
rz(-pi) q[23];
x q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
x q[21];
rz(-pi) q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
rz(-pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[18];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-2.11110676382277) q[18];
sx q[18];
rz(1.50114550081575) q[18];
sx q[18];
rz(4.93453058413595) q[18];
rz(-1.30555142758492) q[21];
sx q[21];
rz(2.07972548140241) q[21];
sx q[21];
rz(5.24509539497965) q[21];
rz(-4.98943679400179) q[23];
sx q[23];
rz(2.7813647328249) q[23];
sx q[23];
rz(6.35647162037376) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
measure q[18] -> c[0];
measure q[21] -> c[1];
measure q[23] -> c[2];
