OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4785[3];
rz(-1.0690806) q[0];
sx q[0];
rz(-2.3738876) q[0];
sx q[0];
rz(-1.7915134) q[0];
rz(3.9942269) q[1];
sx q[1];
rz(5.8465863) q[1];
sx q[1];
rz(11.144614) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[1];
x q[1];
rz(1.6336188) q[2];
sx q[2];
rz(-0.12152306) q[2];
sx q[2];
rz(-1.9815526) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
rz(-pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-1.3500793) q[0];
sx q[0];
rz(-2.3738876) q[0];
sx q[0];
rz(1.0690806) q[0];
rz(1.9815526) q[1];
sx q[1];
rz(0.12152306) q[1];
sx q[1];
rz(1.7198361) q[2];
sx q[2];
rz(-2.7049936) q[2];
sx q[2];
rz(-0.85263427) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4785[0];
measure q[1] -> c4785[1];
measure q[2] -> c4785[2];
