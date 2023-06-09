OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4703[3];
rz(0.84353998) q[0];
sx q[0];
rz(-2.8362032) q[0];
sx q[0];
rz(2.6437824) q[0];
rz(5.8692501) q[1];
sx q[1];
rz(5.721164) q[1];
sx q[1];
rz(15.163662) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[0];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
rz(-2.2754085) q[2];
sx q[2];
rz(-1.4875939) q[2];
sx q[2];
rz(1.214157) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(0.49781027) q[0];
sx q[0];
rz(-2.8362032) q[0];
sx q[0];
rz(-0.84353998) q[0];
rz(-1.9274357) q[1];
sx q[1];
rz(-1.6539988) q[1];
sx q[1];
rz(-0.54430125) q[2];
sx q[2];
rz(-2.5795714) q[2];
sx q[2];
rz(-2.7276575) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4703[0];
measure q[1] -> c4703[1];
measure q[2] -> c4703[2];
