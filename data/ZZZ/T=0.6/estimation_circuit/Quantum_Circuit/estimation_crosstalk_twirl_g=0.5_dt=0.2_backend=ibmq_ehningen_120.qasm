OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg c[3];
rz(6.10133183516143) q[18];
sx q[18];
rz(4.70149153756779) q[18];
sx q[18];
rz(10.727937769735) q[18];
rz(3.32202108007673) q[21];
sx q[21];
rz(4.11094390976134) q[21];
sx q[21];
rz(11.1891840836104) q[21];
rz(5.85466273240243) q[23];
sx q[23];
rz(3.88501304739354) q[23];
sx q[23];
rz(10.0355997103879) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[18];
sx q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[21];
rz(pi/2) q[23];
sx q[23];
rz(-pi) q[23];
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
rz(-pi/2) q[18];
sx q[18];
rz(-pi/2) q[18];
x q[21];
rz(pi) q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
sx q[18];
rz(-pi/2) q[18];
rz(-pi) q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
sx q[18];
rz(-pi) q[21];
x q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
sx q[18];
rz(-pi) q[18];
rz(-pi) q[21];
x q[21];
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
rz(-pi) q[18];
rz(-pi) q[21];
x q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi/2) q[18];
rz(-pi) q[21];
x q[21];
rz(-pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
x q[21];
rz(pi/2) q[23];
sx q[23];
rz(-pi) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[21];
x q[21];
rz(pi/2) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[21];
rz(-pi/2) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[21];
x q[21];
rz(-pi) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[18];
x q[21];
rz(-pi/2) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi) q[21];
rz(pi) q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(pi/2) q[18];
sx q[18];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi/2) q[18];
x q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi/2) q[18];
rz(pi) q[23];
barrier q[18],q[23],q[21];
cx q[23],q[21];
barrier q[18],q[23],q[21];
rz(-pi) q[18];
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
rz(pi/2) q[18];
x q[21];
barrier q[18],q[23],q[21];
cx q[23],q[21];
barrier q[18],q[23],q[21];
rz(-pi) q[18];
sx q[18];
sx q[23];
rz(-pi) q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[21];
x q[21];
rz(-pi/2) q[23];
sx q[23];
barrier q[23],q[18],q[21];
cx q[18],q[21];
barrier q[23],q[18],q[21];
rz(-pi) q[18];
rz(-pi) q[21];
x q[21];
rz(pi/2) q[23];
sx q[23];
rz(-pi/2) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[21];
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
x q[21];
sx q[23];
rz(-pi) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi/2) q[18];
sx q[18];
rz(-pi/2) q[18];
rz(-pi) q[21];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi) q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi/2) q[18];
sx q[18];
rz(-pi/2) q[18];
x q[21];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi/2) q[18];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi) q[18];
x q[21];
x q[23];
barrier q[18],q[23],q[21];
cx q[23],q[21];
barrier q[18],q[23],q[21];
rz(-pi/2) q[18];
sx q[18];
x q[23];
barrier q[18],q[21],q[23];
cx q[21],q[23];
barrier q[18],q[21],q[23];
rz(-pi) q[18];
sx q[18];
x q[21];
x q[23];
rz(-pi/2) q[23];
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
x q[18];
rz(-pi) q[21];
x q[21];
rz(pi/2) q[23];
sx q[23];
rz(-pi/2) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-1.30315980896567) q[18];
sx q[18];
rz(1.5816937696118) q[18];
sx q[18];
rz(3.32344612560795) q[18];
rz(-0.610821749618543) q[21];
sx q[21];
rz(2.39817225978604) q[21];
sx q[21];
rz(3.57011522836695) q[21];
rz(-1.764406122841) q[23];
sx q[23];
rz(2.17224139741824) q[23];
sx q[23];
rz(6.10275688069265) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
measure q[18] -> c[0];
measure q[21] -> c[1];
measure q[23] -> c[2];
