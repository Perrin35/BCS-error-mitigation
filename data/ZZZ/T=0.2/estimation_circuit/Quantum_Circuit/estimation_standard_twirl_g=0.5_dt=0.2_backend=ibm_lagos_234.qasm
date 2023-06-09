OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4734[3];
rz(5.7491604) q[0];
sx q[0];
rz(4.2573699) q[0];
sx q[0];
rz(14.651451) q[0];
rz(0.91864819) q[1];
sx q[1];
rz(-0.065116186) q[1];
sx q[1];
rz(2.5305602) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[0];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[0];
x q[1];
rz(1.5417117) q[2];
sx q[2];
rz(-0.74902862) q[2];
sx q[2];
rz(-2.9867501) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[1];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[1];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[0];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(2.0850808) q[0];
sx q[0];
rz(-1.1157773) q[0];
sx q[0];
rz(-2.6075678) q[0];
rz(-0.15484258) q[1];
sx q[1];
rz(-0.74902862) q[1];
sx q[1];
rz(0.61103244) q[2];
sx q[2];
rz(-0.065116186) q[2];
sx q[2];
rz(-0.91864819) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4734[0];
measure q[1] -> c4734[1];
measure q[2] -> c4734[2];
