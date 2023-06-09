OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4626[3];
rz(-0.29829919) q[0];
sx q[0];
rz(-1.902772) q[0];
sx q[0];
rz(-2.4193328) q[0];
rz(4.8414197) q[1];
sx q[1];
rz(4.4044863) q[1];
sx q[1];
rz(9.8896501) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[0];
rz(0.56167506) q[2];
sx q[2];
rz(-3.0049996) q[2];
sx q[2];
rz(-1.5774487) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
rz(-pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[1];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
rz(-pi) q[2];
x q[2];
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
rz(2.4193328) q[0];
sx q[0];
rz(-1.902772) q[0];
sx q[0];
rz(0.29829919) q[0];
rz(1.5641439) q[1];
sx q[1];
rz(0.1365931) q[1];
sx q[1];
rz(-0.46487209) q[2];
sx q[2];
rz(-1.878699) q[2];
sx q[2];
rz(1.4417656) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4626[0];
measure q[1] -> c4626[1];
measure q[2] -> c4626[2];
