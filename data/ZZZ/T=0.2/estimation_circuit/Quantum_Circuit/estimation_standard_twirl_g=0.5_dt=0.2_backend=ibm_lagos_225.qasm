OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4725[3];
rz(2.2541253) q[0];
sx q[0];
rz(-1.4373172) q[0];
sx q[0];
rz(1.676653) q[0];
rz(-2.6443955) q[1];
sx q[1];
rz(-2.1674635) q[1];
sx q[1];
rz(-0.83119181) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
rz(0.62242408) q[2];
sx q[2];
rz(-1.6547991) q[2];
sx q[2];
rz(0.33263389) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
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
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(1.4649396) q[0];
sx q[0];
rz(-1.4373172) q[0];
sx q[0];
rz(-2.2541253) q[0];
rz(2.8089588) q[1];
sx q[1];
rz(-1.6547991) q[1];
sx q[1];
rz(-2.3104008) q[2];
sx q[2];
rz(-2.1674635) q[2];
sx q[2];
rz(2.6443955) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4725[0];
measure q[1] -> c4725[1];
measure q[2] -> c4725[2];