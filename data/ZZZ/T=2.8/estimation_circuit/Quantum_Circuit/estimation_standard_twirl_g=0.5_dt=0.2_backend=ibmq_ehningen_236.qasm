OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg c[3];
rz(5.92153391198502) q[18];
sx q[18];
rz(4.5416638598532) q[18];
sx q[18];
rz(11.5419342488725) q[18];
rz(5.17328012889985) q[21];
sx q[21];
rz(3.20391700509136) q[21];
sx q[21];
rz(9.42488514065514) q[21];
rz(4.61127380892896) q[23];
sx q[23];
rz(6.17054084400827) q[23];
sx q[23];
rz(14.3518960339546) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[18];
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
x q[18];
rz(-pi) q[21];
x q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[21];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
rz(-pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
x q[21];
rz(-pi) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[18];
x q[21];
rz(pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
rz(-pi) q[23];
x q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
rz(-pi) q[21];
rz(-pi) q[23];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
x q[18];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[18];
rz(-pi) q[21];
x q[21];
rz(-pi) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[18];
x q[18];
rz(-pi) q[21];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
x q[18];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[21];
x q[21];
rz(pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[21];
x q[21];
rz(pi) q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
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
x q[18];
rz(-pi) q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[18];
x q[18];
rz(pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[18];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[21];
rz(pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
rz(-pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[18];
x q[18];
rz(pi) q[21];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
rz(-pi) q[23];
x q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
rz(-pi) q[21];
rz(-pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
x q[21];
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
x q[18];
rz(-pi) q[21];
x q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[18];
x q[18];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[18];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
x q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[18];
rz(pi) q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
x q[21];
rz(-pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
x q[21];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
rz(-pi) q[21];
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
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[18];
rz(pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[21];
x q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[21];
rz(pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
x q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[21];
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
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
x q[18];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
x q[18];
rz(-pi) q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[21];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
x q[21];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
x q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[18];
rz(-pi) q[23];
x q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
rz(-pi) q[21];
x q[21];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
x q[21];
rz(-pi) q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
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
rz(-pi) q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[18];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[18];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[21];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[18];
x q[21];
rz(pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
x q[21];
rz(-pi) q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
rz(-pi) q[21];
x q[21];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
x q[18];
rz(-pi) q[21];
rz(-pi) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[18];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
x q[18];
rz(-pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[21];
x q[21];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
x q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[18];
x q[18];
rz(-pi) q[21];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
rz(-pi) q[21];
rz(-pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
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
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[18];
rz(pi) q[21];
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
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[21];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
x q[21];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[18];
x q[18];
rz(-pi) q[23];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
x q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
x q[21];
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
x q[18];
rz(-pi) q[21];
rz(-pi) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
x q[18];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
x q[18];
x q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[21];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
x q[21];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
x q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
rz(-pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[18];
rz(-pi) q[21];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
x q[18];
rz(-pi) q[21];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[18];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
x q[18];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
x q[21];
rz(-pi) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[18];
x q[21];
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
rz(-pi) q[21];
rz(-pi) q[23];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
x q[18];
x q[21];
rz(-pi) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[18];
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
x q[18];
rz(-pi) q[21];
x q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
rz(-pi) q[23];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[18];
x q[18];
rz(-pi) q[23];
x q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
rz(-pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
x q[21];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[18];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
rz(-pi) q[21];
rz(-pi) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[18];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[23];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
x q[21];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[21];
x q[21];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
x q[18];
rz(-pi) q[21];
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
rz(-pi) q[18];
rz(-pi) q[21];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[18];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[23];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
x q[21];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[18];
rz(-pi) q[21];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
rz(-pi) q[21];
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
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
x q[18];
rz(-pi) q[21];
rz(-pi) q[23];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-2.11715628810309) q[18];
sx q[18];
rz(1.74152144732639) q[18];
sx q[18];
rz(3.50324404878436) q[18];
rz(-0.000107179885765002) q[21];
sx q[21];
rz(3.07926830208822) q[21];
sx q[21];
rz(4.25149783186953) q[21];
rz(-4.92711807318518) q[23];
sx q[23];
rz(0.112644463171315) q[23];
sx q[23];
rz(4.81350415184042) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
measure q[18] -> c[0];
measure q[21] -> c[1];
measure q[23] -> c[2];