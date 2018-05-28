/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 50;

	// add noise
	normal_distribution<double> norm_dist_x(x, std[0]);
	normal_distribution<double> norm_dist_y(y, std[1]);
	normal_distribution<double> norm_dist_theta(theta, std[2]);

	for (int i = 0; i < num_particles; i++) {
		particles.push_back(Particle { i, norm_dist_x(gen), norm_dist_y(gen), norm_dist_theta(gen), 1.0 });
		weights.push_back(1.0);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	normal_distribution<double> norm_dist_x(0, std_pos[0]);
	normal_distribution<double> norm_dist_y(0, std_pos[1]);
	normal_distribution<double> norm_dist_theta(0, std_pos[2]);

	for (Particle& particle : particles) {

		if (fabs(yaw_rate) < AVOID_DIV_BY_ZERO) {
			particle.x += velocity * delta_t * cos(particle.theta);
			particle.y += velocity * delta_t * sin(particle.theta);
		} else {
			particle.x += (velocity / yaw_rate) * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
			particle.y += (velocity / yaw_rate) * (cos(particle.theta) - cos(particle.theta + (yaw_rate * delta_t)));
			particle.theta += yaw_rate * delta_t;
		}

		particle.x += norm_dist_x(gen);
		particle.y += norm_dist_y(gen);
		particle.theta += norm_dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (LandmarkObs& obs : observations) {

		double min_dist = numeric_limits<double>::max();
		int map_id = -1;

		for (const LandmarkObs& pred : predicted) {
			double curr_dist = dist(obs.x, obs.y, pred.x, pred.y);

			if (curr_dist < min_dist) {
				min_dist = curr_dist;
				map_id = pred.id;
			}
		}
		obs.id = map_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	double x_p;
	double y_p;
	double theta_p;

	for (int i = 0; i < num_particles; ++i) {

		Particle& particle = particles[i];

		vector<LandmarkObs> predictions;
		vector<LandmarkObs> obs_transformed;

		x_p = particle.x;
		y_p = particle.y;
		theta_p = particle.theta;

		for (const Map::single_landmark_s& landmark : map_landmarks.landmark_list) {
			int id_i = landmark.id_i;
			float y_f = landmark.y_f;
			float x_f = landmark.x_f;

			if (dist(x_f, y_f, x_p, y_p) <= sensor_range) {
				predictions.push_back(LandmarkObs { id_i, x_f, y_f });
			}
		}		

		for (const LandmarkObs& obs : observations) {
			double t_x = x_p + cos(theta_p) * obs.x - sin(theta_p) * obs.y;
			double t_y = y_p + sin(theta_p) * obs.x + cos(theta_p) * obs.y;

			obs_transformed.push_back(LandmarkObs { obs.id, t_x, t_y });
		}

		dataAssociation(predictions, obs_transformed);

		particle.weight = 1.0;

		for (const LandmarkObs& obs : obs_transformed) {
			double x_pred, y_pred;

			for (const LandmarkObs& prediction : predictions) {
				if (prediction.id == obs.id) {
					y_pred = prediction.y;
					x_pred = prediction.x;
					break;
				}
			}

			double std_x = std_landmark[0];
			double std_y = std_landmark[1];
			double w = 1.0 / (2.0 * M_PI * std_x * std_y) * exp( -pow(x_pred - obs.x, 2) 
				/ (2 * std_x * std_x) - pow(y_pred - obs.y, 2)
				/ (2 * std_y * std_y));

			particle.weight *= w;
		}

		weights[i] = particle.weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	std::discrete_distribution<int> discrete_dist(weights.begin(), weights.end());
	vector<Particle> resampled_particles;

	for (int i = 0; i < num_particles; ++i) {
		int index = discrete_dist(gen);
		resampled_particles.push_back(particles[index]);
	}

	particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
