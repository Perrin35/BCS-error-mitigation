OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg c[3];
rz(2.20961118544386) q[18];
sx q[18];
rz(4.88187984934051) q[18];
sx q[18];
rz(10.8553968496111) q[18];
rz(4.06489704504846) q[21];
sx q[21];
rz(5.76870652167031) q[21];
sx q[21];
rz(13.5745248331884) q[21];
rz(0.901601129800965) q[23];
sx q[23];
rz(6.24265045629367) q[23];
sx q[23];
rz(13.8962716298821) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[18];
x q[21];
rz(-pi) q[23];
sx q[23];
rz(-pi) q[23];
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
rz(-pi) q[18];
x q[18];
rz(-pi) q[21];
rz(-pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[18];
rz(pi/2) q[18];
x q[21];
x q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi) q[18];
x q[18];
rz(-pi) q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi/2) q[18];
rz(-pi) q[21];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
sx q[18];
rz(-pi) q[18];
x q[21];
x q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi) q[18];
sx q[18];
rz(pi/2) q[18];
rz(-pi) q[21];
x q[21];
barrier q[18],q[23],q[21];
cx q[23],q[21];
barrier q[18],q[23],q[21];
rz(pi/2) q[18];
sx q[18];
rz(-pi) q[18];
x q[21];
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
rz(-pi) q[21];
x q[21];
rz(-pi/2) q[23];
sx q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[18];
x q[18];
rz(-pi) q[21];
rz(-pi/2) q[23];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[18];
rz(pi) q[21];
sx q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi/2) q[23];
sx q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[18];
rz(-pi) q[21];
rz(-pi/2) q[23];
sx q[23];
rz(-pi/2) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi) q[21];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
x q[18];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(pi/2) q[18];
sx q[18];
rz(-pi/2) q[18];
rz(-pi) q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[18];
sx q[18];
rz(-pi) q[18];
rz(-pi) q[23];
x q[23];
barrier q[18],q[23],q[21];
cx q[23],q[21];
barrier q[18],q[23],q[21];
sx q[18];
rz(pi/2) q[18];
rz(-pi) q[21];
x q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi) q[18];
sx q[18];
rz(pi/2) q[18];
rz(-pi) q[21];
x q[21];
barrier q[18],q[23],q[21];
cx q[23],q[21];
barrier q[18],q[23],q[21];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(-pi) q[23];
sx q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[18];
x q[18];
rz(-pi) q[21];
x q[21];
rz(pi/2) q[23];
sx q[23];
rz(-pi) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[18];
x q[18];
x q[21];
rz(pi/2) q[23];
sx q[23];
rz(-pi/2) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[21];
rz(-pi) q[23];
sx q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[18];
rz(-pi) q[21];
rz(-pi/2) q[23];
sx q[23];
rz(-pi) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[18];
rz(pi/2) q[18];
x q[21];
rz(pi) q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(pi/2) q[18];
sx q[18];
rz(-pi) q[18];
x q[21];
x q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi) q[18];
sx q[18];
rz(-pi) q[18];
rz(-pi) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi/2) q[18];
rz(pi) q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
sx q[18];
rz(-pi/2) q[18];
rz(-pi) q[21];
rz(-pi) q[23];
x q[23];
barrier q[18],q[23],q[21];
cx q[23],q[21];
barrier q[18],q[23],q[21];
rz(-pi) q[18];
rz(-pi) q[21];
x q[21];
x q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(pi/2) q[18];
sx q[18];
rz(-pi/2) q[18];
rz(-pi) q[21];
x q[21];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[18];
x q[21];
rz(-pi) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[18];
rz(-pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[18];
x q[18];
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
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[18];
rz(-pi) q[21];
rz(-pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
sx q[18];
rz(-pi) q[18];
rz(-pi) q[21];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi) q[18];
sx q[18];
rz(pi/2) q[18];
rz(-pi) q[21];
x q[21];
x q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi/2) q[18];
x q[18];
rz(-pi) q[23];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi) q[21];
rz(pi) q[23];
barrier q[18],q[23],q[21];
cx q[23],q[21];
barrier q[18],q[23],q[21];
sx q[18];
rz(-pi/2) q[18];
x q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi/2) q[18];
sx q[18];
rz(-pi) q[18];
rz(-pi) q[21];
barrier q[18],q[23],q[21];
cx q[23],q[21];
barrier q[18],q[23],q[21];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
x q[21];
rz(-pi) q[23];
sx q[23];
rz(-pi) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[21];
x q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[21];
sx q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-1.43061888884174) q[18];
sx q[18];
rz(1.40130545783908) q[18];
sx q[18];
rz(7.21516677532552) q[18];
rz(-4.14974687241902) q[21];
sx q[21];
rz(0.514478785509278) q[21];
sx q[21];
rz(5.35988091572092) q[21];
rz(-4.47149366911273) q[23];
sx q[23];
rz(0.040534850885912) q[23];
sx q[23];
rz(8.52317683096841) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
measure q[18] -> c[0];
measure q[21] -> c[1];
measure q[23] -> c[2];