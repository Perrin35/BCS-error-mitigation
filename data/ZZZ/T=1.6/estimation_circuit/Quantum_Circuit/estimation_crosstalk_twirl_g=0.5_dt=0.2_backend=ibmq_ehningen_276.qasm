OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg c[3];
rz(1.52253604270181) q[18];
sx q[18];
rz(6.02589599273221) q[18];
sx q[18];
rz(11.1439619365411) q[18];
rz(4.44678406345814) q[21];
sx q[21];
rz(5.14086104149006) q[21];
sx q[21];
rz(12.5648434671761) q[21];
rz(4.5104792674336) q[23];
sx q[23];
rz(5.24709034209914) q[23];
sx q[23];
rz(15.403123100081) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[21];
x q[21];
rz(pi/2) q[23];
sx q[23];
rz(-pi/2) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[18];
rz(-pi) q[21];
rz(-pi) q[23];
x q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
x q[21];
rz(pi/2) q[23];
sx q[23];
rz(-pi/2) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[18];
sx q[18];
rz(-pi) q[18];
rz(pi) q[21];
rz(pi) q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(pi/2) q[18];
sx q[18];
rz(-pi) q[18];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(-pi) q[21];
rz(-pi) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
sx q[18];
rz(-pi) q[18];
rz(-pi) q[21];
x q[21];
rz(pi) q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(pi/2) q[18];
sx q[18];
x q[21];
rz(-pi) q[23];
barrier q[18],q[23],q[21];
cx q[23],q[21];
barrier q[18],q[23],q[21];
sx q[18];
rz(pi/2) q[18];
rz(-pi) q[21];
x q[21];
x q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi) q[18];
sx q[18];
rz(-pi) q[21];
x q[21];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[21];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[18];
rz(-pi) q[21];
rz(-pi/2) q[23];
sx q[23];
rz(-pi/2) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[18];
rz(-pi) q[21];
x q[21];
rz(pi/2) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[18];
x q[18];
rz(-pi) q[23];
sx q[23];
rz(-pi/2) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[18];
rz(-pi) q[21];
rz(-pi/2) q[23];
sx q[23];
rz(-pi/2) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[18];
rz(pi/2) q[18];
rz(pi) q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi) q[18];
x q[21];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
x q[18];
rz(-pi/2) q[18];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi/2) q[18];
x q[21];
rz(pi) q[23];
barrier q[18],q[23],q[21];
cx q[23],q[21];
barrier q[18],q[23],q[21];
rz(-pi/2) q[18];
sx q[18];
rz(-pi) q[18];
rz(-pi) q[21];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
sx q[18];
rz(-pi/2) q[18];
x q[23];
barrier q[18],q[23],q[21];
cx q[23],q[21];
barrier q[18],q[23],q[21];
rz(-pi/2) q[18];
x q[18];
sx q[23];
rz(-pi) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[21];
sx q[23];
rz(-pi/2) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[18];
x q[18];
rz(-pi) q[21];
x q[21];
rz(pi/2) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[18];
rz(-pi) q[21];
x q[21];
rz(-pi/2) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[18];
x q[18];
rz(-pi) q[21];
x q[21];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
x q[21];
rz(pi/2) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[18];
sx q[18];
rz(-pi) q[18];
rz(-pi) q[21];
x q[21];
rz(pi) q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi/2) q[18];
sx q[18];
rz(-pi) q[21];
x q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(pi/2) q[18];
sx q[18];
rz(-pi/2) q[18];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[18];
rz(-pi/2) q[18];
rz(-pi) q[21];
x q[21];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi) q[23];
barrier q[18],q[23],q[21];
cx q[23],q[21];
barrier q[18],q[23],q[21];
rz(-pi) q[18];
sx q[18];
rz(pi/2) q[18];
rz(-pi) q[23];
x q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(-pi) q[21];
rz(pi/2) q[23];
sx q[23];
rz(-pi/2) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[18];
rz(-pi) q[21];
x q[21];
rz(-pi/2) q[23];
sx q[23];
rz(-pi) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
x q[21];
rz(pi/2) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[21];
rz(pi/2) q[23];
sx q[23];
rz(-pi/2) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[18];
x q[18];
rz(-pi) q[21];
sx q[23];
rz(pi/2) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
x q[18];
rz(-pi) q[21];
rz(-pi) q[23];
sx q[23];
rz(-pi) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi/2) q[18];
sx q[18];
rz(-pi/2) q[18];
rz(-pi) q[21];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi) q[18];
x q[18];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(pi/2) q[18];
sx q[18];
rz(-pi/2) q[18];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[18];
sx q[18];
rz(-pi) q[18];
rz(pi) q[23];
barrier q[18],q[23],q[21];
cx q[23],q[21];
barrier q[18],q[23],q[21];
rz(-pi/2) q[18];
sx q[18];
rz(-pi) q[23];
x q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
sx q[18];
rz(pi/2) q[18];
x q[21];
barrier q[18],q[23],q[21];
cx q[23],q[21];
barrier q[18],q[23],q[21];
rz(-pi) q[18];
sx q[18];
rz(pi/2) q[23];
sx q[23];
rz(-pi/2) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
x q[21];
rz(-pi) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
x q[18];
x q[21];
rz(-pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[18];
x q[21];
x q[23];
rz(-pi/2) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
x q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[18];
x q[21];
rz(-pi/2) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[18];
rz(-pi/2) q[18];
x q[21];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
x q[18];
x q[21];
rz(-pi) q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi/2) q[18];
rz(-pi) q[21];
rz(-pi) q[23];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi/2) q[18];
sx q[18];
rz(-pi/2) q[18];
rz(-pi) q[21];
x q[21];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi) q[18];
rz(-pi) q[21];
x q[21];
rz(-pi) q[23];
barrier q[18],q[23],q[21];
cx q[23],q[21];
barrier q[18],q[23],q[21];
rz(-pi) q[18];
rz(-pi) q[23];
x q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(-pi) q[21];
rz(-pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[21];
x q[21];
rz(-pi/2) q[23];
sx q[23];
rz(-pi) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(pi/2) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[18];
rz(-pi) q[21];
x q[21];
sx q[23];
rz(-pi) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
x q[18];
rz(-pi) q[21];
x q[21];
rz(-pi) q[23];
sx q[23];
rz(-pi) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi/2) q[18];
sx q[18];
rz(-pi/2) q[18];
rz(pi) q[21];
x q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(pi/2) q[18];
sx q[18];
rz(-pi) q[21];
rz(-pi) q[23];
x q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
x q[18];
rz(-pi/2) q[18];
rz(-pi) q[21];
rz(-pi) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
sx q[18];
rz(-pi) q[18];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[18],q[23],q[21];
cx q[23],q[21];
barrier q[18],q[23],q[21];
sx q[18];
rz(-pi/2) q[18];
rz(-pi) q[21];
rz(-pi) q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi/2) q[18];
sx q[18];
rz(-pi) q[18];
x q[21];
rz(-pi) q[23];
barrier q[18],q[23],q[21];
cx q[23],q[21];
barrier q[18],q[23],q[21];
rz(-pi) q[18];
sx q[18];
rz(-pi) q[18];
rz(-pi) q[21];
rz(-pi/2) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[21];
x q[21];
rz(pi/2) q[23];
sx q[23];
rz(-pi) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
x q[18];
rz(-pi) q[21];
x q[21];
rz(-pi) q[23];
sx q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[18];
rz(pi) q[21];
sx q[23];
rz(-pi) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[18];
x q[18];
rz(-pi) q[21];
rz(-pi) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
x q[21];
rz(-pi) q[23];
sx q[23];
rz(-pi) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
sx q[18];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(pi/2) q[18];
sx q[18];
rz(-pi) q[18];
x q[21];
rz(-pi) q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(-pi) q[21];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[18];
rz(-pi/2) q[18];
rz(pi) q[21];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi) q[18];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[18],q[23],q[21];
cx q[23],q[21];
barrier q[18],q[23],q[21];
rz(pi/2) q[18];
sx q[18];
rz(-pi) q[18];
rz(-pi) q[21];
x q[21];
x q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
sx q[18];
rz(-pi) q[18];
rz(-pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi/2) q[23];
sx q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[18];
x q[18];
x q[21];
rz(-pi/2) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[18];
x q[18];
sx q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[21];
x q[21];
rz(-pi) q[23];
sx q[23];
rz(-pi/2) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
x q[18];
rz(-pi) q[21];
x q[21];
rz(-pi/2) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[18];
sx q[18];
rz(pi) q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(pi/2) q[18];
sx q[18];
rz(-pi) q[18];
rz(-pi) q[21];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi/2) q[18];
sx q[18];
rz(-pi/2) q[18];
rz(-pi) q[21];
rz(-pi) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi/2) q[18];
sx q[18];
rz(-pi/2) q[18];
x q[21];
x q[23];
barrier q[18],q[23],q[21];
cx q[23],q[21];
barrier q[18],q[23],q[21];
rz(-pi/2) q[18];
sx q[18];
rz(-pi) q[18];
x q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
sx q[18];
rz(-pi/2) q[18];
rz(-pi) q[21];
x q[21];
barrier q[18],q[23],q[21];
cx q[23],q[21];
barrier q[18],q[23],q[21];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(-pi/2) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[18];
rz(pi/2) q[23];
sx q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[18];
rz(-pi) q[21];
x q[21];
rz(-pi) q[23];
sx q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-1.71918397577174) q[18];
sx q[18];
rz(0.257289314447372) q[18];
sx q[18];
rz(7.90224191806757) q[18];
rz(-3.14006550640674) q[21];
sx q[21];
rz(1.14232426568953) q[21];
sx q[21];
rz(4.97799389731124) q[21];
rz(-5.97834513931162) q[23];
sx q[23];
rz(1.03609496508045) q[23];
sx q[23];
rz(4.91429869333578) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
measure q[18] -> c[0];
measure q[21] -> c[1];
measure q[23] -> c[2];
