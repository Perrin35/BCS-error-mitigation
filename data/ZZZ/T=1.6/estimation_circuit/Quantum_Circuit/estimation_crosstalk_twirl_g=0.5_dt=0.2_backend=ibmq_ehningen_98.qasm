OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg c[3];
rz(4.17045185061474) q[18];
sx q[18];
rz(5.47071573206541) q[18];
sx q[18];
rz(12.463006582617) q[18];
rz(0.390609649591707) q[21];
sx q[21];
rz(4.73129480715561) q[21];
sx q[21];
rz(13.6143669539963) q[21];
rz(3.88487149463316) q[23];
sx q[23];
rz(5.49165177713027) q[23];
sx q[23];
rz(13.1485055572983) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[21];
rz(-pi) q[23];
sx q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
x q[18];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
x q[18];
x q[21];
rz(-pi) q[23];
sx q[23];
rz(-pi) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[18];
rz(pi/2) q[18];
rz(pi) q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
x q[21];
x q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi/2) q[18];
x q[18];
x q[21];
rz(-pi) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[18];
rz(-pi/2) q[18];
rz(-pi) q[21];
x q[21];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi) q[18];
rz(-pi) q[21];
barrier q[18],q[23],q[21];
cx q[23],q[21];
barrier q[18],q[23],q[21];
rz(pi/2) q[18];
sx q[18];
x q[21];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi) q[18];
sx q[18];
rz(-pi) q[21];
x q[21];
sx q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[18];
sx q[23];
rz(-pi/2) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[18];
x q[18];
rz(-pi) q[21];
x q[23];
rz(-pi/2) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[23];
rz(pi/2) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[18];
x q[18];
x q[21];
sx q[23];
rz(-pi/2) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[18];
x q[18];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi) q[21];
rz(pi) q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
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
rz(-pi) q[18];
x q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[18];
rz(pi/2) q[18];
rz(pi) q[21];
barrier q[18],q[23],q[21];
cx q[23],q[21];
barrier q[18],q[23],q[21];
x q[18];
rz(-pi) q[21];
rz(-pi) q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(pi/2) q[18];
sx q[18];
x q[21];
barrier q[18],q[23],q[21];
cx q[23],q[21];
barrier q[18],q[23],q[21];
sx q[18];
rz(-pi) q[18];
rz(-pi) q[21];
x q[21];
x q[23];
rz(-pi/2) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
x q[18];
rz(-pi) q[21];
rz(-pi) q[23];
sx q[23];
rz(pi/2) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
x q[21];
rz(-pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[18];
rz(pi) q[21];
rz(-pi) q[23];
sx q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[18];
x q[18];
rz(-pi) q[21];
x q[21];
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
rz(pi/2) q[18];
x q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi) q[18];
sx q[18];
rz(-pi/2) q[18];
rz(-pi) q[21];
x q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
sx q[18];
rz(-pi) q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi) q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(pi/2) q[18];
sx q[18];
rz(-pi) q[18];
rz(-pi) q[21];
rz(-pi) q[23];
x q[23];
barrier q[18],q[23],q[21];
cx q[23],q[21];
barrier q[18],q[23],q[21];
sx q[18];
rz(-pi/2) q[18];
x q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(-pi) q[21];
x q[21];
rz(-pi/2) q[23];
sx q[23];
rz(-pi/2) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
x q[18];
x q[21];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[18];
rz(-pi) q[21];
rz(pi/2) q[23];
sx q[23];
rz(-pi/2) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[18];
x q[18];
rz(pi) q[21];
rz(-pi) q[23];
sx q[23];
rz(-pi) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[18];
x q[18];
x q[21];
rz(-pi) q[23];
sx q[23];
rz(-pi/2) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[21];
rz(-pi/2) q[23];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
sx q[18];
rz(pi) q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
sx q[18];
rz(pi/2) q[18];
x q[21];
x q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi/2) q[18];
x q[18];
x q[21];
rz(-pi) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
sx q[18];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[18],q[23],q[21];
cx q[23],q[21];
barrier q[18],q[23],q[21];
sx q[18];
rz(pi/2) q[18];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
x q[18];
rz(-pi) q[21];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[18],q[23],q[21];
cx q[23],q[21];
barrier q[18],q[23],q[21];
rz(pi/2) q[18];
x q[21];
rz(-pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[18];
rz(-pi) q[21];
rz(pi/2) q[23];
sx q[23];
rz(-pi) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
x q[21];
rz(pi/2) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[18];
rz(pi) q[21];
rz(-pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
x q[18];
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
sx q[18];
rz(-pi) q[18];
x q[21];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
x q[18];
x q[21];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi) q[18];
sx q[18];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[18];
sx q[18];
rz(-pi) q[18];
rz(pi) q[21];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi) q[18];
x q[18];
rz(-pi) q[21];
rz(-pi) q[23];
x q[23];
barrier q[18],q[23],q[21];
cx q[23],q[21];
barrier q[18],q[23],q[21];
sx q[18];
rz(-pi/2) q[18];
rz(-pi) q[21];
x q[21];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
x q[18];
rz(-pi/2) q[18];
rz(-pi) q[21];
x q[23];
rz(-pi/2) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[18];
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
sx q[23];
rz(-pi/2) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[18];
rz(pi) q[21];
rz(-pi/2) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
x q[18];
rz(-pi) q[21];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
x q[18];
x q[23];
rz(-pi/2) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
sx q[18];
rz(pi/2) q[18];
rz(-pi) q[21];
rz(-pi) q[23];
x q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
sx q[18];
rz(-pi) q[18];
x q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi/2) q[18];
rz(-pi) q[21];
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
rz(-pi) q[18];
sx q[18];
rz(pi/2) q[18];
rz(-pi) q[21];
x q[21];
barrier q[18],q[23],q[21];
cx q[23],q[21];
barrier q[18],q[23],q[21];
sx q[18];
x q[21];
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
rz(-pi) q[21];
rz(pi/2) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[18];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[18];
x q[18];
rz(-pi) q[21];
x q[21];
rz(-pi/2) q[23];
sx q[23];
rz(-pi) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[21];
rz(-pi/2) q[23];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[18];
sx q[18];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi) q[18];
rz(-pi) q[21];
x q[21];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
sx q[18];
rz(-pi) q[21];
rz(-pi) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[18];
rz(-pi/2) q[18];
rz(pi) q[21];
rz(-pi) q[23];
x q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
sx q[18];
rz(pi/2) q[18];
rz(-pi) q[23];
barrier q[18],q[23],q[21];
cx q[23],q[21];
barrier q[18],q[23],q[21];
x q[18];
x q[21];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(-pi) q[21];
x q[21];
rz(pi/2) q[23];
sx q[23];
rz(-pi/2) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
x q[18];
rz(-pi/2) q[23];
sx q[23];
rz(-pi) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
x q[18];
rz(-pi) q[21];
rz(-pi/2) q[23];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[21];
x q[21];
x q[23];
rz(-pi/2) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[18];
x q[18];
rz(-pi) q[23];
sx q[23];
rz(pi/2) q[23];
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
rz(-pi) q[18];
sx q[18];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
sx q[18];
rz(-pi/2) q[18];
rz(-pi) q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi/2) q[18];
x q[18];
rz(-pi) q[21];
rz(-pi) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi/2) q[18];
x q[21];
barrier q[18],q[23],q[21];
cx q[23],q[21];
barrier q[18],q[23],q[21];
sx q[18];
rz(-pi/2) q[18];
rz(-pi) q[23];
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
x q[18];
rz(-pi/2) q[18];
sx q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(pi/2) q[23];
sx q[23];
rz(-pi) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[18];
x q[18];
x q[21];
rz(-pi/2) q[23];
sx q[23];
rz(-pi/2) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-3.03822862184758) q[18];
sx q[18];
rz(0.812469575114179) q[18];
sx q[18];
rz(5.25432611015464) q[18];
rz(-4.18958899322689) q[21];
sx q[21];
rz(1.55189050002397) q[21];
sx q[21];
rz(9.03416831117767) q[21];
rz(-3.72372759652893) q[23];
sx q[23];
rz(0.791533530049317) q[23];
sx q[23];
rz(5.53990646613622) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
measure q[18] -> c[0];
measure q[21] -> c[1];
measure q[23] -> c[2];
