OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4685[3];
rz(3.1101331) q[0];
sx q[0];
rz(-1.2515335) q[0];
sx q[0];
rz(-1.5253929) q[0];
rz(-1.3661656) q[1];
sx q[1];
rz(-2.3325472) q[1];
sx q[1];
rz(-0.87560456) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[1];
rz(-0.6693066) q[2];
sx q[2];
rz(-2.8390338) q[2];
sx q[2];
rz(1.4094772) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-1.6161997) q[0];
sx q[0];
rz(-1.2515335) q[0];
sx q[0];
rz(-3.1101331) q[0];
rz(1.7321154) q[1];
sx q[1];
rz(-2.8390338) q[1];
sx q[1];
rz(-0.87560456) q[2];
sx q[2];
rz(-0.8090454) q[2];
sx q[2];
rz(-1.775427) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4685[0];
measure q[1] -> c4685[1];
measure q[2] -> c4685[2];