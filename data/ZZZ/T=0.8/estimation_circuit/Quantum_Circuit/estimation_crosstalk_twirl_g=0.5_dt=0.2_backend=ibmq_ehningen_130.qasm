OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg c[3];
rz(1.13434868188393) q[18];
sx q[18];
rz(5.04099270637078) q[18];
sx q[18];
rz(12.8293095737345) q[18];
rz(5.80869705477838) q[21];
sx q[21];
rz(5.58437415373347) q[21];
sx q[21];
rz(13.7114119379698) q[21];
rz(2.34068957832201) q[23];
sx q[23];
rz(4.87358933832155) q[23];
sx q[23];
rz(13.0958442956202) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[21];
rz(-pi) q[23];
sx q[23];
rz(-pi) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[18];
x q[18];
rz(-pi) q[21];
x q[21];
rz(-pi/2) q[23];
sx q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
x q[18];
rz(-pi) q[21];
x q[21];
rz(pi/2) q[23];
sx q[23];
rz(-pi/2) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[18];
rz(-pi/2) q[18];
rz(pi) q[21];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi) q[18];
x q[18];
rz(-pi) q[21];
x q[21];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(pi/2) q[18];
x q[21];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[18];
rz(pi/2) q[18];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi/2) q[18];
sx q[18];
rz(-pi) q[21];
x q[23];
barrier q[18],q[23],q[21];
cx q[23],q[21];
barrier q[18],q[23],q[21];
rz(-pi/2) q[18];
sx q[18];
rz(-pi) q[21];
x q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(-pi) q[23];
sx q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
x q[18];
x q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[18];
x q[18];
x q[21];
sx q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[18];
x q[18];
rz(pi) q[21];
x q[23];
rz(pi/2) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[21];
x q[21];
rz(pi/2) q[23];
sx q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
x q[18];
x q[21];
sx q[23];
rz(-pi) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
sx q[18];
rz(-pi) q[18];
rz(-pi) q[23];
x q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi/2) q[18];
sx q[18];
rz(-pi) q[18];
x q[21];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
x q[21];
rz(-pi) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[18];
rz(pi/2) q[18];
x q[23];
barrier q[18],q[23],q[21];
cx q[23],q[21];
barrier q[18],q[23],q[21];
rz(-pi/2) q[18];
sx q[18];
rz(-pi) q[18];
rz(-pi) q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
sx q[18];
rz(-pi/2) q[18];
rz(-pi) q[21];
x q[23];
barrier q[18],q[23],q[21];
cx q[23],q[21];
barrier q[18],q[23],q[21];
rz(pi/2) q[18];
rz(-pi) q[21];
x q[21];
rz(-pi) q[23];
sx q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
x q[18];
sx q[23];
rz(pi/2) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[18];
rz(-pi) q[21];
rz(-pi/2) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[18];
x q[18];
rz(-pi) q[21];
x q[21];
rz(-pi) q[23];
sx q[23];
rz(-pi) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
x q[18];
rz(-pi) q[21];
rz(-pi) q[23];
sx q[23];
rz(-pi/2) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi/2) q[23];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi/2) q[18];
sx q[18];
rz(-pi/2) q[18];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi) q[18];
sx q[18];
rz(pi/2) q[18];
x q[21];
x q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
sx q[18];
rz(-pi) q[18];
rz(-pi) q[23];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
sx q[18];
rz(-pi) q[18];
x q[21];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi) q[18];
x q[23];
barrier q[18],q[23],q[21];
cx q[23],q[21];
barrier q[18],q[23],q[21];
rz(-pi) q[23];
x q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
sx q[18];
rz(-pi) q[18];
rz(-pi) q[21];
rz(-pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[18];
rz(-pi) q[23];
sx q[23];
rz(-pi/2) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
x q[18];
x q[21];
sx q[23];
rz(-pi) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[18];
rz(-pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[18];
sx q[23];
rz(pi/2) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[18];
x q[18];
sx q[23];
rz(-pi) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi/2) q[18];
sx q[18];
rz(-pi/2) q[18];
x q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi) q[21];
rz(-pi) q[23];
x q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(-pi) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
sx q[18];
rz(-pi) q[18];
rz(pi) q[21];
rz(pi) q[23];
barrier q[18],q[23],q[21];
cx q[23],q[21];
barrier q[18],q[23],q[21];
rz(pi/2) q[18];
sx q[18];
rz(-pi) q[18];
x q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
x q[18];
rz(-pi) q[21];
x q[21];
barrier q[18],q[23],q[21];
cx q[23],q[21];
barrier q[18],q[23],q[21];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(-pi) q[21];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
x q[18];
x q[21];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[21];
rz(-pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-3.40453161296508) q[18];
sx q[18];
rz(1.2421926008088) q[18];
sx q[18];
rz(8.29042927888545) q[18];
rz(-4.2866339772004) q[21];
sx q[21];
rz(0.698811153446117) q[21];
sx q[21];
rz(3.616080905991) q[21];
rz(-3.67106633485078) q[23];
sx q[23];
rz(1.40959596885803) q[23];
sx q[23];
rz(7.08408838244737) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
measure q[18] -> c[0];
measure q[21] -> c[1];
measure q[23] -> c[2];
