OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4741[3];
rz(-0.26476338) q[0];
sx q[0];
rz(-1.4164413) q[0];
sx q[0];
rz(-1.7384645) q[0];
rz(-2.6044791) q[1];
sx q[1];
rz(-2.3897139) q[1];
sx q[1];
rz(0.24213632) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
rz(-pi) q[1];
x q[1];
rz(2.5444719) q[2];
sx q[2];
rz(-0.13949652) q[2];
sx q[2];
rz(-0.66917608) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[1];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(1.7384645) q[0];
sx q[0];
rz(-1.4164413) q[0];
sx q[0];
rz(0.26476338) q[0];
rz(2.4724166) q[1];
sx q[1];
rz(-3.0020961) q[1];
sx q[1];
rz(2.8994563) q[2];
sx q[2];
rz(-2.3897139) q[2];
sx q[2];
rz(2.6044791) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4741[0];
measure q[1] -> c4741[1];
measure q[2] -> c4741[2];
