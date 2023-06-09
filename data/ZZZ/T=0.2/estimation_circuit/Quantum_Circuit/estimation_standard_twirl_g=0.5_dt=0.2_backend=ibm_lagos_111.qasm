OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4611[3];
rz(2.0020442) q[0];
sx q[0];
rz(-1.8302187) q[0];
sx q[0];
rz(0.18158561) q[0];
rz(-0.44946524) q[1];
sx q[1];
rz(-2.2375378) q[1];
sx q[1];
rz(-3.0306668) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[0];
rz(-pi) q[1];
x q[1];
rz(4.2419641) q[2];
sx q[2];
rz(3.994437) q[2];
sx q[2];
rz(15.364854) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[2];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
x q[1];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(2.960007) q[0];
sx q[0];
rz(-1.8302187) q[0];
sx q[0];
rz(-2.0020442) q[0];
rz(0.34310955) q[1];
sx q[1];
rz(2.2887483) q[1];
sx q[1];
rz(-3.0306668) q[2];
sx q[2];
rz(-0.90405485) q[2];
sx q[2];
rz(-2.6921274) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4611[0];
measure q[1] -> c4611[1];
measure q[2] -> c4611[2];
